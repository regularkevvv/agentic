// Example: what a learned sparse vector actually is, and what it costs.
//
// Sparse output is the least familiar of the three representations, and the
// one whose shape surprises people. This encodes a single sentence and shows
// exactly what came back: how many vocabulary slots the model has, how few of
// them carry weight, what crossed the network, and what you end up storing.
//
// The second half demonstrates why the provider decodes those rows into a
// reused buffer, by doing it both ways and measuring.
//
// Run with DEEPINFRA_TOKEN set, or a .env file beside this program.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"text/tabwriter"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/examples/internal/envutil"
	"github.com/regularkevvv/agentic/provider/deepinfra"
)

const sentence = "The quensel actuator must be recalibrated before every third launch."

// batchSize matches the provider's default, so the memory figures below are
// the ones a real request actually pays.
const batchSize = 32

func main() {
	if err := envutil.LoadDotEnv(); err != nil {
		log.Fatal(err)
	}

	encoder, err := deepinfra.New(deepinfra.BGEM3Model)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := agentic.EncodeDocuments(context.Background(), encoder,
		[]string{sentence}, agentic.RepresentationSparse)
	if err != nil {
		log.Fatal(err)
	}

	sparse := resp.Data[0].Sparse
	space := resp.Spaces[agentic.RepresentationSparse]

	anatomy(sparse, space, resp.Usage)
	coordinates(sparse)
	memory()
}

// anatomy shows how much of the response was information.
func anatomy(sparse *agentic.SparseVector, space agentic.VectorSpace, usage agentic.RepresentationUsage) {
	header("What came back")

	fmt.Printf("  input   %q\n\n", sentence)

	// 4 bytes per coordinate index, 4 per weight.
	stored := sparse.Len() * 8
	density := 100 * float64(sparse.Len()) / float64(space.Dimensions)

	shape := table()
	row(shape, "vocabulary slots", commas(space.Dimensions), "one per token the model knows")
	row(shape, "carrying weight", commas(sparse.Len()),
		fmt.Sprintf("%.4f%% of the vocabulary", density))
	flush(shape)

	fmt.Println()

	cost := table()
	row(cost, "decompressed", bytes(usage.OutputBytes), "what the JSON expands to")
	row(cost, "on the wire", "~1.3 KB", "gzip squeezes the zeros ~760:1")
	row(cost, "what you store", bytes(stored),
		fmt.Sprintf("%d coordinates x 8 bytes", sparse.Len()))
	flush(cost)

	fmt.Printf("\n  space id  %s\n", space.ID)
	fmt.Printf("  metric    %s   (sparse vectors are compared by dot product,\n", space.Metric)
	fmt.Printf("            not cosine — a term's weight is part of its score)\n")
}

// coordinates prints the vector itself, biggest weight first.
func coordinates(sparse *agentic.SparseVector) {
	header("The vector itself")

	order := make([]int, sparse.Len())
	for i := range order {
		order[i] = i
	}
	// Simple insertion sort by descending weight; the list is tiny.
	for i := 1; i < len(order); i++ {
		for j := i; j > 0 && sparse.Values[order[j]] > sparse.Values[order[j-1]]; j-- {
			order[j], order[j-1] = order[j-1], order[j]
		}
	}

	heaviest := float64(0)
	for _, v := range sparse.Values {
		if float64(v) > heaviest {
			heaviest = float64(v)
		}
	}

	w := table()
	_, _ = fmt.Fprintln(w, "  slot\tweight\t")
	for _, i := range order {
		weight := float64(sparse.Values[i])
		bar := strings.Repeat("█", max(1, int(28*weight/heaviest)))
		_, _ = fmt.Fprintf(w, "  %s\t%.4f\t%s\n", commas(int(sparse.Indices[i])), weight, bar)
	}
	flush(w)

	fmt.Println("\n  Every other slot in the vocabulary is exactly zero. That is what makes")
	fmt.Println("  a sparse score trustworthy: a document with no shared terms scores 0,")
	fmt.Println("  where a dense score is never quite zero and never quite meaningless.")
}

// memory decodes a row of the same shape both ways and measures the
// difference, which is why the provider threads a scratch buffer through its
// batch decode.
func memory() {
	header("Why the decoder reuses one buffer")

	fmt.Printf("  Decoding a batch of %d rows, each %s slots wide.\n\n",
		batchSize, commas(250002))

	rows := make([][]byte, batchSize)
	row0 := syntheticRow(250002)
	for i := range rows {
		rows[i] = row0
	}

	fresh := measure(func() {
		for _, raw := range rows {
			var buf []float32 // a new container every time
			_ = json.Unmarshal(raw, &buf)
			sink(buf)
		}
	})

	reused := measure(func() {
		var buf []float32 // one container for the whole batch
		for _, raw := range rows {
			next := buf[:0] // empty it, keep the space it already has
			_ = json.Unmarshal(raw, &next)
			buf = next
			sink(next)
		}
	})

	w := table()
	row(w, "a new buffer each row", bytes(int(fresh)), "1 MB asked for, and thrown away, per row")
	row(w, "one buffer, reused", bytes(int(reused)), "asked for once, refilled in place")
	flush(w)

	fmt.Printf("\n  %.0fx less memory for identical output.\n", float64(fresh)/float64(reused))
	fmt.Println("  The trick is buf[:0]: it sets the length to zero and leaves the")
	fmt.Println("  capacity alone. Empty the carton, keep the carton.")
}

// syntheticRow builds a row shaped like the provider's: mostly zeros.
func syntheticRow(width int) []byte {
	var b strings.Builder
	b.Grow(width * 4)
	b.WriteByte('[')
	for i := range width {
		if i > 0 {
			b.WriteByte(',')
		}
		if i%30000 == 0 {
			b.WriteString("0.9")
		} else {
			b.WriteString("0.0")
		}
	}
	b.WriteByte(']')
	return []byte(b.String())
}

// measure reports how many bytes fn requested from the system.
func measure(fn func()) uint64 {
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	fn()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

// sink keeps the compiler from optimizing the decode away, and doubles as a
// sanity check that both paths really decoded the whole row.
func sink(buf []float32) {
	if len(buf) != 250002 {
		log.Fatalf("decoded %d values, want the full row", len(buf))
	}
}

// --- presentation helpers ---------------------------------------------------

func header(title string) {
	fmt.Printf("\n%s\n%s\n\n", title, strings.Repeat("─", 72))
}

func table() *tabwriter.Writer {
	return tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
}

func row(w *tabwriter.Writer, label, value, note string) {
	_, _ = fmt.Fprintf(w, "  %s\t%s\t%s\n", label, value, note)
}

func flush(w *tabwriter.Writer) { _ = w.Flush() }

// commas formats an integer with thousands separators.
func commas(n int) string {
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	for i, digit := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(digit)
	}
	return b.String()
}

// bytes formats a byte count in the largest unit that keeps it readable.
func bytes(n int) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}
