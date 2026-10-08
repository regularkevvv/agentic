package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

// synthesize speaks text with OpenAI's speech endpoint and returns 24 kHz
// mono PCM16. It stands in for a person talking into the microphone.
func synthesize(ctx context.Context, key, text string) ([]int16, error) {
	body, err := json.Marshal(map[string]string{
		"model": "gpt-4o-mini-tts", "voice": "alloy", "input": text, "response_format": "pcm",
	})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/audio/speech", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Bearer "+key)
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	raw, err := io.ReadAll(io.LimitReader(response.Body, 32<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("synthesize speech: %s: %s", response.Status, raw)
	}
	pcm := make([]int16, len(raw)/2)
	if err := binary.Read(bytes.NewReader(raw[:len(pcm)*2]), binary.LittleEndian, pcm); err != nil {
		return nil, err
	}
	return pcm, nil
}

// mulawFrames resamples 24 kHz PCM16 to 8 kHz and encodes it as 20 ms G.711
// μ-law frames, the format the gateway negotiates with the browser.
func mulawFrames(pcm24k []int16) [][]byte {
	var encoded []byte
	for i := 0; i+2 < len(pcm24k); i += 3 {
		average := (int32(pcm24k[i]) + int32(pcm24k[i+1]) + int32(pcm24k[i+2])) / 3
		encoded = append(encoded, mulawEncode(int16(average)))
	}
	var frames [][]byte
	for len(encoded) > 0 {
		n := min(frameBytes, len(encoded))
		f := append([]byte(nil), encoded[:n]...)
		for len(f) < frameBytes {
			f = append(f, 0xFF)
		}
		frames = append(frames, f)
		encoded = encoded[n:]
	}
	return frames
}

// mulawEncode is ITU-T G.711 μ-law compression of one sample.
func mulawEncode(sample int16) byte {
	const bias, clip = 0x84, 32635
	s := int32(sample)
	sign := byte(0)
	if s < 0 {
		sign, s = 0x80, -s
	}
	s = min(s, clip) + bias
	exponent := byte(7)
	for mask := int32(0x4000); s&mask == 0 && exponent > 0; mask >>= 1 {
		exponent--
	}
	mantissa := byte(s>>(exponent+3)) & 0x0F
	return ^(sign | exponent<<4 | mantissa)
}

// writeWAV stores PCM16 as a mono WAV file, the form Chrome accepts as a fake
// microphone.
func writeWAV(path string, pcm []int16, rate uint32) error {
	var buf bytes.Buffer
	size := uint32(len(pcm) * 2)
	for _, v := range []any{
		[]byte("RIFF"), 36 + size, []byte("WAVE"),
		[]byte("fmt "), uint32(16), uint16(1), uint16(1), rate, rate * 2, uint16(2), uint16(16),
		[]byte("data"), size, pcm,
	} {
		if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	return os.WriteFile(path, buf.Bytes(), 0o600)
}
