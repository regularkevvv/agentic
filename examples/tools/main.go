// Example: agent with auto-registered tool.
package main

import (
	"context"
	"fmt"
	"log"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/examples/internal/envutil"
	"github.com/regularkevvv/agentic/provider/openai"
)

// GetWeatherInput defines the tool schema via struct tags.
// Name is auto-inferred: GetWeatherInput -> "get_weather".
// Description comes from the `tool` tag on the blank struct{} field.
type GetWeatherInput struct {
	_        struct{} `tool:"Look up the current weather for a city"`
	Location string   `json:"location" description:"City and country"`
	Unit     string   `json:"unit,omitempty" description:"Temperature unit" enum:"celsius,fahrenheit"`
}

type WeatherOutput struct {
	Location    string `json:"location"`
	Temperature int    `json:"temperature"`
	Unit        string `json:"unit"`
	Condition   string `json:"condition"`
}

func getWeather(_ context.Context, input GetWeatherInput) (WeatherOutput, error) {
	unit := input.Unit
	if unit == "" {
		unit = "celsius"
	}
	temp := 24
	if unit == "fahrenheit" {
		temp = 75
	}
	return WeatherOutput{
		Location:    input.Location,
		Temperature: temp,
		Unit:        unit,
		Condition:   "clear skies",
	}, nil
}

func main() {
	if err := envutil.LoadDotEnv(); err != nil {
		log.Fatal(err)
	}

	model, err := openai.New("gpt-4o")
	if err != nil {
		log.Fatal(err)
	}

	agent := agentic.NewAgent(
		"You are a helpful travel assistant.",
		model,
	)
	agentic.AddTool(agent, getWeather)

	result, err := agent.Run(
		context.Background(),
		envutil.PromptFromArgs("What's the weather in Lima?"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Output)
}
