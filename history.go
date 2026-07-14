package agentic

import "encoding/json"

// MessageHistory is a serializable conversation history that can be
// saved, loaded, and used to resume conversations.
type MessageHistory struct {
	Messages []Message `json:"messages"`
}

// NewHistory creates a new MessageHistory from the given messages.
func NewHistory(messages ...Message) *MessageHistory {
	return &MessageHistory{Messages: messages}
}

// LoadHistory deserializes a MessageHistory from JSON bytes.
func LoadHistory(data []byte) (*MessageHistory, error) {
	var h MessageHistory
	if err := json.Unmarshal(data, &h); err != nil {
		return nil, err
	}
	return &h, nil
}

// SaveJSON serializes the MessageHistory to JSON bytes.
func (h *MessageHistory) SaveJSON() ([]byte, error) {
	return json.Marshal(h)
}

// ToRunOption converts the history into a RunOption that provides the
// messages as conversation history for a new run.
func (h *MessageHistory) ToRunOption() RunOption {
	return WithMessages(h.Messages...)
}
