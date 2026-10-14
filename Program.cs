using System.Diagnostics;
using Practica4.Rag;

Console.WriteLine("=== Práctica 4 · RAG (embeddings + búsqueda semántica) ===");

// 1. Indexar docs/ (con cache en disco para no re-embedear en CPU en cada arranque)
using var emb = new Embeddings();
var store = new VectorStore();
Console.WriteLine("Indexando docs/...");
var swIndex = Stopwatch.StartNew();
await store.IndexarAsync("docs", "embeddings-cache.json", emb);
swIndex.Stop();
Console.WriteLine($"  Índice listo: {store.Count} chunks [{swIndex.Elapsed.TotalSeconds:F1}s]\n");

using var rag = new RagClient(store, emb);

// 2. Demos: una respondible con docs y una trampa (fuera de docs -> debe decir que no sabe)
await ProcesarAsync("¿Cuántos días tengo para devolver un producto?");
await ProcesarAsync("¿Venden repuestos para licuadoras?");

while (true)
{
    Console.Write("Pregunta > ");
    var input = Console.ReadLine()?.Trim();
    if (string.IsNullOrWhiteSpace(input)) continue;
    if (input.Equals("/salir", StringComparison.OrdinalIgnoreCase)) break;
    if (input.StartsWith("/modelo ", StringComparison.OrdinalIgnoreCase))
    {
        rag.ChatModel = input["/modelo ".Length..].Trim();
        Console.WriteLine($"(modelo chat -> {rag.ChatModel})\n");
        continue;
    }

    await ProcesarAsync(input);
}

Console.WriteLine("Chau!");

async Task ProcesarAsync(string pregunta)
{
    try
    {
        var sw = Stopwatch.StartNew();
        var (respuesta, usados) = await rag.PreguntarAsync(pregunta);
        sw.Stop();
        Console.WriteLine("  Recuperados:");
        foreach (var (h, i) in usados.Select((h, i) => (h, i)))
            Console.WriteLine($"    [{i + 1}] {h.Chunk.Doc} (score {h.Score:F3}): {Recorte(h.Chunk.Texto)}");
        Console.WriteLine($"  IA > {respuesta} [{sw.Elapsed.TotalSeconds:F1}s]\n");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  [error] {ex.Message}\n");
    }
}

static string Recorte(string s) => s.Length <= 90 ? s.Replace('\n', ' ') : s[..90].Replace('\n', ' ') + "…";
