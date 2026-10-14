using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Practica4.Rag;

// RAG: pregunta -> embedding -> top-3 chunks -> prompt con contexto -> chat responde SOLO con ese contexto.
public sealed class RagClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly VectorStore _store;
    private readonly Embeddings _emb;

    public string ChatModel { get; set; }

    public RagClient(VectorStore store, Embeddings emb, string baseUrl = "http://localhost:11434", string chatModel = "qwen2.5:1.5b")
    {
        _store = store;
        _emb = emb;
        ChatModel = chatModel;
        _http = new HttpClient { BaseAddress = new Uri(baseUrl), Timeout = TimeSpan.FromMinutes(5) };
    }

    public async Task<(string Respuesta, List<Hit> Usados)> PreguntarAsync(string pregunta, CancellationToken ct = default)
    {
        var qv = (await _emb.EmbedAsync([pregunta], ct))[0];
        var hits = _store.Buscar(qv, topK: 3);

        var contexto = string.Join("\n\n", hits.Select((h, i) => $"[{i + 1}] ({h.Chunk.Doc}) {h.Chunk.Texto}"));
        var messages = new JsonArray
        {
            new JsonObject
            {
                ["role"] = "system",
                ["content"] = "Respondé en español rioplatense, breve, usando ÚNICAMENTE el contexto de abajo. " +
                              "Si la respuesta no está en el contexto, decí exactamente: \"No tengo esa información en mis documentos.\" " +
                              "No inventes datos, plazos ni números."
            },
            new JsonObject { ["role"] = "user", ["content"] = $"Contexto:\n{contexto}\n\nPregunta: {pregunta}" }
        };

        var body = new JsonObject { ["model"] = ChatModel, ["messages"] = messages, ["stream"] = false };
        var response = await _http.PostAsJsonAsync("/api/chat", body, ct);
        response.EnsureSuccessStatusCode();

        using var doc = await response.Content.ReadFromJsonAsync<JsonDocument>(cancellationToken: ct);
        var texto = doc!.RootElement.GetProperty("message").GetProperty("content").GetString()?.Trim() ?? "(vacío)";
        return (texto, hits);
    }

    public void Dispose() => _http.Dispose();
}
