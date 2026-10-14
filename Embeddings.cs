using System.Net.Http.Json;
using System.Text.Json;

namespace Practica4.Rag;

// Traductor texto -> números vía POST /api/embed con nomic-embed-text.
public sealed class Embeddings : IDisposable
{
    private readonly HttpClient _http;
    public string Model { get; }

    public Embeddings(string baseUrl = "http://localhost:11434", string model = "nomic-embed-text")
    {
        _http = new HttpClient { BaseAddress = new Uri(baseUrl), Timeout = TimeSpan.FromMinutes(5) };
        Model = model;
    }

    // Embebe varios textos en un solo request (input acepta arreglo).
    public async Task<List<float[]>> EmbedAsync(IEnumerable<string> textos, CancellationToken ct = default)
    {
        var lista = textos.ToList();
        var body = new { model = Model, input = lista };
        HttpResponseMessage response;
        try
        {
            response = await _http.PostAsJsonAsync("/api/embed", body, ct);
        }
        catch (HttpRequestException ex)
        {
            throw new InvalidOperationException("No pude conectar con Ollama en http://localhost:11434. ¿Está corriendo?", ex);
        }
        response.EnsureSuccessStatusCode();

        using var doc = await response.Content.ReadFromJsonAsync<JsonDocument>(cancellationToken: ct);
        return doc!.RootElement.GetProperty("embeddings").EnumerateArray()
            .Select(e => e.EnumerateArray().Select(n => n.GetSingle()).ToArray())
            .ToList();
    }

    public void Dispose() => _http.Dispose();
}
