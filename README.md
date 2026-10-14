# Práctica 4 — RAG: Embeddings + búsqueda semántica (.NET + Ollama local)

Preguntas sobre documentos propios con citas y sin alucinaciones. Sin NuGets: `HttpClient` puro contra `http://localhost:11434`.

```bash
ollama pull nomic-embed-text   # una sola vez (~274MB)
dotnet run
```

## El flujo RAG (la lección principal)

```
pregunta → embedding → top-3 chunks por coseno → prompt con contexto
→ el chat responde SOLO con ese contexto (o dice que no sabe)
```

## Archivos

- `docs/*.md` — empresa ficticia "NubeSur" (devoluciones, envíos, garantía, horarios). Datos inventados que el modelo no puede saber: prueban que no alucine. Viajan junto al `.exe` (`CopyToOutputDirectory`) para que ande con `dotnet run` y con F5.
- `Embeddings.cs` — `POST /api/embed` con `nomic-embed-text` (traductor texto→números, no conversa).
- `VectorStore.cs` — chunking por párrafos, cache `embeddings-cache.json` por hash SHA256 (en CPU no se re-embebea en cada arranque), búsqueda top-3 por similitud coseno escrita a mano.
- `RagClient.cs` — arma el prompt con contexto + regla anti-alucinación y pide la respuesta a `qwen2.5:1.5b`.
- `Program.cs` — indexa con progreso, 2 demos + loop (`/salir`, `/modelo <nombre>`), muestra fragmentos recuperados con score.

## Conceptos

- **Chunk**: cada párrafo es una unidad con un solo tema; el doc entero mezclaría temas y la búsqueda pierde precisión.
- **Coseno**: ángulo entre vectores (1 = mismo significado). Los scores dicen qué tan cerca; el prompt decide qué hacer.
- **Grounding**: el system prompt obliga a responder solo con el contexto. Retrieval trae candidatos, prompt evita inventos.
- **Citas**: doc + score de cada fragmento usado = auditoría (igual que el `🔧` de la P3).

## Ejemplo

```
  Recuperados:
    [1] devoluciones (score 0.735): Aceptamos devoluciones dentro de los 30 días corridos...
  IA > Tienes 30 días desde la entrega para hacer una devolución. [3.9s]

  Recuperados:
    [1] envios (score 0.636): Repartimos con nuestra flota propia...
  IA > No tengo esa información en mis documentos. [1.5s]
```
