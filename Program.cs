using System.Net.Http;
using System.Text.RegularExpressions;

string apiKey = Environment.GetEnvironmentVariable("GEMINI_API_KEY")
    ?? throw new InvalidOperationException("Missing GEMINI_API_KEY environment variable.");
if (string.IsNullOrWhiteSpace(apiKey))
{
    Console.WriteLine("Missing API key. Set the GEMINI_API_KEY environment variable to your Google AI Studio API key.");
    return;
}

using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
http.DefaultRequestHeaders.Add("x-goog-api-key", apiKey);

Console.WriteLine("Welcome to the game");

int rounds;
while (true)
{
    Console.Write("How many rounds should this game have: ");
    if (int.TryParse(Console.ReadLine(), out rounds) && rounds > 0)
        break;

    Console.WriteLine("Please enter a whole number greater than zero.");
}

Console.WriteLine("Enter the player names separated by commas:");
Console.Write("> ");

List<string> names = (Console.ReadLine() ?? "")
    .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
    .ToList();

if (names.Count == 0)
{
    Console.WriteLine("You must enter at least one player name.");
    return;
}

// Dictionary keys must be unique. Names are treated as case-insensitive.
if (names.Distinct(StringComparer.OrdinalIgnoreCase).Count() != names.Count)
{
    Console.WriteLine("Please enter a unique name for each player.");
    return;
}

Dictionary<string, int> scores = new(StringComparer.OrdinalIgnoreCase);
foreach (string name in names)
{
    scores[name] = 0;
}

try
{
    for (int round = 1; round <= rounds; round++)
    {
        Console.WriteLine($"\n========== Round {round} of {rounds} ==========\n");
        PrintScores(names, scores);

        Console.WriteLine("\nA scenario is being generated...");

        GemmaResult scenario = await Gemma.AskAsync(
            http,
            prompt: "Generate a short deadly scenario that people have to try to escape. Make it no more than 50 words. Do not give players any options in the prompt.");

        Console.WriteLine("\nSCENARIO:");
        Console.WriteLine(scenario.Text);

        if (scenario.FinishReason != "STOP")
            Console.WriteLine($"Warning: response finish reason was {scenario.FinishReason}.");

        List<string> responses = new();

        foreach (string name in names)
        {
            Console.WriteLine($"\n{name}, how will you survive?");
            Console.Write("> ");
            responses.Add(Console.ReadLine() ?? "");
        }

        Console.WriteLine("\n========== THE AI DECIDES YOUR FATE ==========\n");

        for (int i = 0; i < names.Count; i++)
        {
            string fatePrompt =
                $"Decide whether {names[i]} lives or dies based on the following scenario:\n\n" +
                $"{scenario.Text}\n\nTheir response is: {responses[i]}\n\n" +
                "Give a short summary of the course of events. End your response with exactly one of these verdict lines: " +
                "\"Verdict: survived\" or \"Verdict: died\". Do not change the wording of the verdict line.";

            GemmaResult fate = await Gemma.AskAsync(http, fatePrompt);

            Console.WriteLine($"{names[i]},\n{fate.Text}\n");

            // Read the verdict from the generated text, not FinishReason.
            Match verdict = Regex.Match(
                fate.Text,
                @"Verdict:\s*(survived|died)\b",
                RegexOptions.IgnoreCase);

            if (!verdict.Success)
            {
                Console.WriteLine($"Couldn't identify a clear verdict for {names[i]}, so no point was awarded.");
            }
            else if (verdict.Groups[1].Value.Equals("survived", StringComparison.OrdinalIgnoreCase))
            {
                scores[names[i]]++;
                Console.WriteLine($"{names[i]} survived and earns 1 point!");
            }
            else
            {
                Console.WriteLine($"{names[i]} died and earns 0 points this round.");
            }
        }

        Console.WriteLine("\n========== SCORES AFTER ROUND " + round + " ==========");
        PrintScores(names, scores);
    }
}
catch (HttpRequestException ex)
{
    Console.WriteLine($"\nThe API request failed: {ex.Message}");
    Console.WriteLine("Check your API key, model availability, and Google AI Studio status, then try again.");
}
catch (TaskCanceledException)
{
    Console.WriteLine("\nThe request timed out. Please try again.");
}

static void PrintScores(List<string> names, Dictionary<string, int> scores)
{
    Console.WriteLine("Current scores:");
    foreach (string name in names)
    {
        Console.WriteLine($"{name}: {scores[name]} point(s)");
    }
}
