# Архитектура hay_v2_bot

Бот построен из трёх пайплайнов Haystack и тонкого слоя Telegram. Хендлеры не ходят в OpenAI и Pinecone напрямую: они вызывают сервисы (`Generator`, `DocumentIngestor`, `Summarizer`, `ChatMemory`), а сервисы запускают пайплайны. Все сервисы собираются один раз в `main.build_services()`.

## Хранилище

Один индекс Pinecone (dimension 1536, metric cosine) и два namespace:

| Namespace | Что лежит | Кто пишет | Кто читает | Кто очищает |
|---|---|---|---|---|
| `docs` (`PINECONE_NAMESPACE_DOCS`) | чанки файлов | `ingestion_pipeline` | `generation_pipeline.docs_retriever`, `/docs` | `/cleardocs` |
| `chat` (`PINECONE_NAMESPACE_CHAT`) | сообщения пользователя | `ChatMemory.save_user_message` | `generation_pipeline.memory_retriever` | `/clear` |

На каждое namespace создаётся свой `PineconeDocumentStore` (`components/document_store.create_document_store`). Все запросы фильтруются по `meta.user_id`, поэтому пользователи не видят чужие данные.

### Meta чанка документа

| Поле | Тип | Откуда |
|---|---|---|
| `file_name` | str | исходное имя файла из Telegram (очищенное от путей) |
| `chunk_index` | int | порядковый номер непустого чанка, с 0 |
| `page` | int, если есть | `page_number` от Docling или минимальный `prov.page_no` из `dl_meta` |
| `headings` | str, если есть | заголовки раздела из `dl_meta.headings` |
| `user_id`, `chat_id` | str | Telegram |
| `uploaded_at` | str | время загрузки, ISO 8601 UTC |
| `source` | str | всегда `"document"` |

Вложенный `dl_meta` и служебные поля `split_id` и `split_idx_start` удаляются: Pinecone хранит только `str`, `int`, `float`, `bool` и `list[str]`, остальное он отбросил бы с предупреждением.

Id чанка детерминированный: `doc-<sha256(user_id, file_name)[:24]>-<chunk_index>`. Перед записью `DocumentIngestor` удаляет прежние чанки этого файла у этого пользователя. Повторная загрузка заменяет файл, а не дублирует его.

### Meta сообщения в памяти

`user_id`, `chat_id`, `timestamp`, `role="user"`, `source="chat"`. Id сообщения — `<chat_id>-<message_id>`. Сохраняется только непустой текст от человека: команды, нетекстовые сообщения и сообщения ботов не сохраняются.

## ingestion_pipeline

Файл: `pipelines/ingestion_pipeline.py`. Запускается в отдельном потоке (`ThreadPoolExecutor(max_workers=1)`), файлы обрабатываются по очереди.

| Компонент | Класс | Вход | Выход |
|---|---|---|---|
| `converter` | `DoclingConverter(export_type=ExportType.DOC_CHUNKS, chunker=HybridChunker())` | `sources: list[str]`, `meta: dict` | `documents` |
| `meta_enricher` | `ChunkMetaEnricher` (свой) | `documents` | `documents` |
| `embedder` | `OpenAIDocumentEmbedder(model=text-embedding-3-small, api_base_url=…, meta_fields_to_embed=["file_name", "headings"])` | `documents` | `documents`, `meta` |
| `writer` | `DocumentWriter(document_store=docs, policy=DuplicatePolicy.OVERWRITE)` | `documents` | `documents_written` |

Связи: `converter.documents → meta_enricher.documents → embedder.documents → writer.documents`.

Вход при запуске: `{"converter": {"sources": [path], "meta": {file_name, user_id, chat_id, uploaded_at, source}}}`. Docling добавляет переданный `meta` к каждому чанку. `include_outputs_from={"meta_enricher"}` возвращает чанки: по ним строится резюме. Если чанков нет (например, скан без текстового слоя), `DocumentIngestor` поднимает `IngestionError` с понятным пользователю текстом.

## generation_pipeline

Файл: `pipelines/generation_pipeline.py`. Запускается на каждое текстовое сообщение.

| Компонент | Класс | Вход | Выход |
|---|---|---|---|
| `text_embedder` | `OpenAITextEmbedder(model=text-embedding-3-small, api_base_url=…)` | `text` | `embedding`, `meta` |
| `docs_retriever` | `PineconeEmbeddingRetriever(docs, top_k=DOCS_TOP_K)` | `query_embedding`, `filters` | `documents` |
| `docs_threshold` | `ScoreThreshold(min_score=DOCS_MIN_SCORE)` (свой) | `documents` | `documents` |
| `memory_retriever` | `PineconeEmbeddingRetriever(chat, top_k=MEMORY_TOP_K)` | `query_embedding`, `filters` | `documents` |
| `prompt_builder` | `ChatPromptBuilder(template=[user-сообщение], required_variables=["question"])` | `documents`, `memories`, `question` | `prompt` |
| `llm` | `Agent(chat_generator=OpenAIChatGenerator(api_base_url=…), tools=[…], system_prompt=…)` | `messages` | `messages`, `last_message` |

Связи:

```
text_embedder.embedding     → docs_retriever.query_embedding
text_embedder.embedding     → memory_retriever.query_embedding
docs_retriever.documents    → docs_threshold.documents
docs_threshold.documents    → prompt_builder.documents
memory_retriever.documents  → prompt_builder.memories
prompt_builder.prompt       → llm.messages
```

Вход при запуске: текст в `text_embedder.text` и `prompt_builder.question`, фильтр `meta.user_id == <id>` в оба ретривера.

Шаблон промпта помечает каждый фрагмент строкой `[Источник: <file_name>, стр. <page>]`. Системный промпт требует отвечать по фрагментам и в конце писать «Источник: …», а без опоры на фрагменты на файл не ссылаться. `ScoreThreshold` отсекает слабые совпадения, чтобы на вопрос «какая погода» бот не цитировал случайный документ.

**Почему LLM-узел — Agent, а не голый OpenAIChatGenerator.** Бот должен сохранить инструменты из VPg06. `Agent` — это компонент Haystack, который внутри вызывает тот же `OpenAIChatGenerator`, исполняет вызовы инструментов и повторяет запрос к модели, пока не получит текстовый ответ. Его вход `messages` принимает `prompt` от `ChatPromptBuilder` напрямую.

После запуска `Generator` берёт из результата:

- `llm.last_message.text` — ответ пользователю;
- результаты инструмента `dog_image` — ссылки на фото, бот отправляет их через `send_photo`;
- `text_embedder.embedding` — тот же вектор записывается в память (namespace `chat`), второй запрос за эмбеддингом не нужен;
- `docs_threshold.documents` — источники для логов.

Сообщение сохраняется в память **после** поиска, поэтому текущее сообщение не попадает в собственный контекст.

## summary_pipeline

Файл: `pipelines/summary_pipeline.py`.

| Компонент | Класс | Вход | Выход |
|---|---|---|---|
| `prompt_builder` | `ChatPromptBuilder([system, user])` | `file_name`, `text` | `prompt` |
| `llm` | `OpenAIChatGenerator(api_base_url=…, temperature=0.2, max_tokens=120)` | `messages` | `replies` |
| `one_sentence` | `OneSentence` (свой) | `replies` | `summary` |

`text` — первые `SUMMARY_CHUNKS` чанков, всего не больше 6000 символов. Модель просят ответить одним предложением, но `OneSentence` это дополнительно гарантирует: берёт первое предложение, убирает префиксы вроде «Резюме:» и ставит точку, если её нет.

## Инструменты агента

`components/tools.py` — перенесено из VPg06 без изменений поведения.

| Имя | Параметры | Источник |
|---|---|---|
| `cat_fact` | — | catfact.ninja |
| `dog_image` | — | dog.ceo |
| `dog_image_analyzer` | `image_url` | vision-модель через `OpenAIChatGenerator` с `api_base_url` |
| `weather` | `city` | Open-Meteo (геокодинг и прогноз) |

Каждый инструмент обёрнут в `logged_tool`: вызов, аргументы и результат пишутся в лог, а исключение превращается в понятный текст для агента.

## Telegram-слой

`bot/telegram_bot.py`:

- `/start`, `/help` — статичный текст;
- `/clear` — `ChatMemory.clear(user_id)`, только namespace `chat`;
- `/docs` — `docs_store.filter_documents(user_filter)` с группировкой по `file_name` (`chunk_meta.summarize_files`);
- `/cleardocs` — `delete_by_filter(docs_store, user_filter)`;
- текст — `Generator.answer`, затем сохранение в память, фото собак и ответ, разбитый по 4096 символов;
- документ — `bot/files.validate_upload` (расширение и размер; Telegram отдаёт ботам до 20 МБ). Затем ответ «Файл получен…» и задача в поток ingestion: скачать в `tmp/<временная папка>/`, `DocumentIngestor.ingest`, «Готово…», `Summarizer.summarize`, резюме. Временная папка удаляется при выходе из `with`. При ошибке пользователь получает понятное сообщение, а в лог пишется traceback.

## Конфигурация

`config.load_config()` читает `.env.local` из корня проекта. Проверки:

- обязательны `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `PINECONE_API_KEY`, `PINECONE_INDEX`;
- `OPENAI_BASE_URL` начинается с `http://` или `https://`;
- модель эмбеддингов известна, размерность берётся из таблицы;
- namespace документов и памяти различаются;
- числовые параметры — положительные числа.

`config.openai_kwargs()` — единственное место, где собираются `api_key`, `model` и `api_base_url` для всех компонентов OpenAI.

## Логирование

| Событие | Где | Пример строки |
|---|---|---|
| старт ingestion | `DocumentIngestor.ingest` | `Ingestion: старт file=…` |
| число чанков после Docling | `ChunkMetaEnricher` | `Ingestion: Docling вернул чанков=…, непустых=…, страницы=…` |
| запись в Pinecone | `DocumentIngestor.ingest` | `Ingestion: запись в Pinecone file=…, чанков=…, записано=…, время=…` |
| поиск по документам | `ScoreThreshold` | `Поиск по документам: найдено=…, после порога … осталось=…, источники=…` |
| поиск по памяти | `Generator.answer` | `Поиск в Pinecone: user_id=…, память=…, документы=…` |
| запись в память | `ChatMemory` | `Память: записано в Pinecone …` |
| вызовы инструментов | `logged_tool`, `Generator` | `Вызов инструмента weather, аргументы: …` |
| ошибки | хендлеры | `logger.exception(...)` с traceback |
