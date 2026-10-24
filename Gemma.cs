using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;

public sealed record GemmaResult(
    string Text,
    string FinishReason,
    int CandidateCount,
    int? PromptTokenCount,
    int? CandidatesTokenCount);

public static class Gemma
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    private const int MaxAttempts = 3;
    private const string FinalOnlyInstruction =
        "Return only the final answer. Do not include private reasoning, analysis, planning, scratch work, or token counts.";

    public const string DefaultModel = "gemma-4-26b-a4b-it";

    public static async Task<GemmaResult> AskAsync(
        HttpClient http, string prompt, string? systemInstruction = null,
        string model = DefaultModel, CancellationToken cancellationToken = default)
    {
        string fullPrompt = BuildPrompt(prompt, systemInstruction);

        for (int attempt = 1; ; attempt++)
        {
            using var response = await http.PostAsJsonAsync(
                $"{BaseUrlFor(model)}:generateContent",
                BuildRequest(fullPrompt), JsonOpts, cancellationToken);

            if (response.IsSuccessStatusCode)
            {
                using var doc = await JsonDocument.ParseAsync(
                    await response.Content.ReadAsStreamAsync(cancellationToken),
                    cancellationToken: cancellationToken);

                return ParseResult(doc)
                    ?? throw new InvalidOperationException(
                        "Gemma API returned no candidates. The prompt may have been blocked or the response may be incomplete.");
            }

            string body = await response.Content.ReadAsStringAsync(cancellationToken);
            if (attempt >= MaxAttempts || !IsTransient(response.StatusCode))
                throw CreateApiException(response.StatusCode, body);

            await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt - 1)), cancellationToken);
        }
    }

    public static async IAsyncEnumerable<string> AskStreamingAsync(
        HttpClient http, string prompt, string? systemInstruction = null,
        string model = DefaultModel,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        string fullPrompt = BuildPrompt(prompt, systemInstruction);
        HttpResponseMessage? successfulResponse = null;

        for (int attempt = 1; attempt <= MaxAttempts; attempt++)
        {
            using var request = new HttpRequestMessage(
                HttpMethod.Post,
                $"{BaseUrlFor(model)}:streamGenerateContent?alt=sse")
            {
                Content = JsonContent.Create(BuildRequest(fullPrompt), options: JsonOpts)
            };

            var response = await http.SendAsync(
                request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

            if (response.IsSuccessStatusCode)
            {
                successfulResponse = response;
                break;
            }

            string body = await response.Content.ReadAsStringAsync(cancellationToken);
            HttpStatusCode statusCode = response.StatusCode;
            response.Dispose();

            if (attempt == MaxAttempts || !IsTransient(statusCode))
                throw CreateApiException(statusCode, body);

            await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt - 1)), cancellationToken);
        }

        if (successfulResponse is null)
            throw new HttpRequestException("Gemma API request failed without returning a response.");

        using (successfulResponse)
        using (var reader = new StreamReader(
            await successfulResponse.Content.ReadAsStreamAsync(cancellationToken)))
        {
            while (await reader.ReadLineAsync(cancellationToken) is { } line)
            {
                if (!line.StartsWith("data: ", StringComparison.Ordinal))
                    continue;

                string payload = line["data: ".Length..];
                if (payload == "[DONE]")
                    yield break;

                using var doc = JsonDocument.Parse(payload);
                if (ExtractText(doc) is { } text)
                    yield return text;
            }
        }
    }

    private static string BaseUrlFor(string model) =>
        $"https://generativelanguage.googleapis.com/v1beta/models/{model}";

    // Gemma models may not accept the Gemini systemInstruction field, so include
    // the instruction in the user prompt for broader model compatibility.
    private static string BuildPrompt(string prompt, string? systemInstruction)
    {
        string instructions = string.IsNullOrWhiteSpace(systemInstruction)
            ? FinalOnlyInstruction
            : $"{systemInstruction.Trim()}\n\n{FinalOnlyInstruction}";

        return $"Instructions: {instructions}\n\nUser request:\n{prompt}";
    }

    private static object BuildRequest(string prompt) => new
    {
        contents = new[]
        {
            new
            {
                role = "user",
                parts = new[] { new { text = prompt } }
            }
        },
        generationConfig = new { }
    };

    private static bool IsTransient(HttpStatusCode statusCode) =>
        statusCode == HttpStatusCode.RequestTimeout ||
        statusCode == HttpStatusCode.TooManyRequests ||
        statusCode == HttpStatusCode.InternalServerError ||
        statusCode == HttpStatusCode.BadGateway ||
        statusCode == HttpStatusCode.ServiceUnavailable ||
        statusCode == HttpStatusCode.GatewayTimeout;

    private static HttpRequestException CreateApiException(HttpStatusCode statusCode, string body) =>
        new($"Gemma API error {(int)statusCode} ({statusCode}): {body}", null, statusCode);

    private static GemmaResult? ParseResult(JsonDocument doc)
    {
        if (!doc.RootElement.TryGetProperty("candidates", out var candidates) ||
            candidates.ValueKind != JsonValueKind.Array || candidates.GetArrayLength() == 0)
            return null;

        var sb = new StringBuilder();
        string finishReason = "UNKNOWN";

        foreach (var candidate in candidates.EnumerateArray())
        {
            if (candidate.TryGetProperty("content", out var content) &&
                content.TryGetProperty("parts", out var parts) &&
                parts.ValueKind == JsonValueKind.Array)
            {
                foreach (var part in parts.EnumerateArray())
                    AppendVisibleText(sb, part);
            }

            if (candidate.TryGetProperty("finishReason", out var fr) &&
                fr.ValueKind == JsonValueKind.String)
                finishReason = fr.GetString() ?? finishReason;
        }

        int? promptTokens = null;
        int? candidateTokens = null;
        if (doc.RootElement.TryGetProperty("usageMetadata", out var usage))
        {
            if (usage.TryGetProperty("promptTokenCount", out var p) && p.TryGetInt32(out int pValue))
                promptTokens = pValue;
            if (usage.TryGetProperty("candidatesTokenCount", out var c) && c.TryGetInt32(out int cValue))
                candidateTokens = cValue;
        }

        return new GemmaResult(sb.ToString().Trim(), finishReason,
            candidates.GetArrayLength(), promptTokens, candidateTokens);
    }

    private static string? ExtractText(JsonDocument doc)
    {
        if (!doc.RootElement.TryGetProperty("candidates", out var candidates) ||
            candidates.ValueKind != JsonValueKind.Array || candidates.GetArrayLength() == 0)
            return null;

        var sb = new StringBuilder();
        if (candidates[0].TryGetProperty("content", out var content) &&
            content.TryGetProperty("parts", out var parts) && parts.ValueKind == JsonValueKind.Array)
        {
            foreach (var part in parts.EnumerateArray())
                AppendVisibleText(sb, part);
        }

        return sb.Length > 0 ? sb.ToString() : null;
    }

    private static void AppendVisibleText(StringBuilder output, JsonElement part)
    {
        if (part.TryGetProperty("thought", out var thought) &&
            thought.ValueKind == JsonValueKind.True)
            return;

        if (part.TryGetProperty("text", out var text) && text.ValueKind == JsonValueKind.String)
            output.Append(text.GetString());
    }
}
