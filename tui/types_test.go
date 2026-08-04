package tui

import "testing"

func TestInputAndResolutionValidation(t *testing.T) {
	t.Parallel()
	if (Input{}).Validate() == nil || (Input{Text: " \n"}).Validate() == nil || (Input{Text: "ok"}).Validate() != nil {
		t.Fatal("unexpected input validation")
	}
	valid := Resolution{SuspensionID: "s", Decisions: []Decision{{CallID: "one", Action: DecisionApprove}, {CallID: "two", Action: DecisionDeny}}, Prompt: &Input{Text: "continue"}}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	cases := []Resolution{
		{},
		{SuspensionID: "s", Decisions: []Decision{{Action: DecisionApprove}}},
		{SuspensionID: "s", Decisions: []Decision{{CallID: "one", Action: DecisionApprove}, {CallID: "one", Action: DecisionDeny}}},
		{SuspensionID: "s", Decisions: []Decision{{CallID: "one", Action: "maybe"}}},
		{SuspensionID: "s", Prompt: &Input{}},
	}
	for index, value := range cases {
		if value.Validate() == nil {
			t.Fatalf("case %d unexpectedly valid", index)
		}
	}
}

func TestUsageCacheHitPercent(t *testing.T) {
	t.Parallel()
	if got := (Usage{}).CacheHitPercent(); got != 0 {
		t.Fatalf("empty hit rate = %v", got)
	}
	if got := (Usage{PromptTokens: 100}).CacheHitPercent(); got != 0 {
		t.Fatalf("zero-read hit rate = %v", got)
	}
	if got := (Usage{PromptTokens: 200, CacheReadTokens: 150}).CacheHitPercent(); got != 75 {
		t.Fatalf("hit rate = %v", got)
	}
}

func TestToolPresenterFunc(t *testing.T) {
	t.Parallel()
	tool := Tool{Name: "read_file"}
	if got := (ToolPresenterFunc(nil)).PresentTool(tool); got != (ToolPresentation{}) {
		t.Fatalf("nil presenter = %#v", got)
	}
	presenter := ToolPresenterFunc(func(value Tool) ToolPresentation {
		return ToolPresentation{Category: ToolCategoryExplore, Title: value.Name}
	})
	if got := presenter.PresentTool(tool); got.Title != "read_file" || got.Category != ToolCategoryExplore {
		t.Fatalf("presentation = %#v", got)
	}
}
