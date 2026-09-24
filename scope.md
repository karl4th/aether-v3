# Aether Stage 1 — Scope

## Статус документа

Этот документ фиксирует цели и границы новой итерации Stage 1.

- Рабочая ветка: `stage1`
- Базовая ветка: `master`
- Обучающий датасет: `manifestro/stage1_aether`
- Заявленный объём обучающих данных: 2 500 часов английской речи
- Внутренний формат датасета рассматривается отдельно перед интеграцией

## 1. Цель Stage 1

Stage 1 должен обучить `AetherSpeech` извлекать из semantic-кодов Mimi
устойчивое представление английской речи, которое:

1. сохраняет лингвистическое содержание;
2. работает на неизвестных голосах;
3. устойчиво к акцентам, различным микрофонам и умеренному шуму;
4. распознаёт короткую разговорную речь, а не только длинное чистое чтение;
5. подходит как frozen speech encoder для Stage 2;
6. остаётся пригодным для последующих direct speech-to-speech стадий.

Целевой тракт:

```text
audio
  → frozen Mimi semantic stream
  → trainable AetherSpeech
  → reusable continuous speech representation
```

CTC-ветка является обучающим и диагностическим инструментом. Итоговый
артефакт Stage 1 — `AetherSpeech`, а не CTC-декодер.

Текущий production run обучается только по CTC. Future-q0 prediction
остаётся в коде как отдельная ablation, но отключена: польза
предсказания того же codebook-0 не доказана, а главная цель этой
итерации — максимально сильное распознавание речи.

## 2. Причина повторного Stage 1

Текущий Stage 1 достиг 16.35% WER на чистой аудиокнижной речи и подтвердил,
что AetherSpeech способен извлекать слова из semantic-кодов Mimi. Этот
результат не подтвердил устойчивость к реальным пользователям.

Наблюдались критические ошибки на неизвестном голосе и короткой разговорной
фразе, включая повторяющийся вывод вида:

```text
WHY ARE YOU
→ WH WH WH WH WH
```

Новая итерация должна улучшить переносимость между говорящими и доменами, а
не только снизить средний WER на одном чистом корпусе.

## 3. Обучающий контракт

Источником данных является `manifestro/stage1_aether`.

Перед использованием требуется отдельно прочитать и проверить фактическую
документацию, схему, splits, provenance и статистику опубликованной ревизии.
Этот scope не предполагает конкретной внутренней структуры датасета.

Обязательные требования к эксперименту:

- Mimi остаётся frozen;
- используется semantic stream Mimi;
- train, validation и test не пересекаются;
- test не используется для настройки или выбора checkpoint;
- источник и ревизия датасета фиксируются в provenance запуска;
- преобразование текста в CTC targets выполняется training pipeline;
- CTC targets не считаются обязательной частью канонического датасета.

## 4. Обучаемые компоненты

Базовый эксперимент сохраняет текущую архитектуру:

| Компонент | Состояние |
|---|---|
| Mimi | frozen |
| AetherSpeech | trainable |
| CTC upsampler | trainable diagnostic branch |
| Byte-level CTC head | trainable diagnostic branch |
| Future-q0 prediction heads | disabled; ablation only |

Текущая итерация не меняет одновременно Mimi и размер AetherSpeech.
Базовый objective — byte-level CTC на всех 2 500 часах.

CTC baseline использует:

- byte-level targets;
- CTC upsampler ×4;
- greedy decoding;
- без внешней языковой модели;
- без LM rescoring;
- без beam search для основной исследовательской метрики.

## 5. Основные критерии качества

### 5.1 Минимальный проходной уровень

На полном независимом test split:

```text
WER < 15%
```

Целевой сильный результат:

```text
WER < 13%
```

Одного общего WER недостаточно для принятия модели.

### 5.2 Неизвестные голоса

Модель должна работать на говорящих, отсутствовавших в train split.

- speaker overlap между train и evaluation запрещён, если metadata позволяет
  его определить;
- результаты для неизвестных голосов считаются отдельно;
- средняя метрика не должна скрывать полный провал отдельных групп.

### 5.3 Акценты и условия записи

Evaluation должен включать доступные срезы по:

- native и non-native English;
- акцентам;
- чистой и шумной речи;
- near-field и far-field записи;
- различным источникам или доменам;
- мужским и женским голосам, если metadata доступна.

### 5.4 Короткие разговорные фразы

Отдельно оцениваются фразы продолжительностью примерно 1–5 секунд, включая
вопросы и команды, характерные для голосового интерфейса.

Минимальная цель:

```text
short-query WER < 15%
```

Желаемая цель:

```text
short-query WER < 10%
```

### 5.5 Катастрофические ошибки

Помимо WER/CER, обязательны:

- empty hypothesis rate;
- repetition-collapse rate;
- utterance WER ≥ 100% rate;
- invalid UTF-8 rate;
- hypothesis/reference length ratio;
- truncated hypothesis rate;
- keyword recall для коротких запросов.

Проходные границы:

```text
repetition-collapse rate < 0.5%
empty hypothesis rate < 0.5%
```

Catastrophic failure rate должен быть существенно ниже текущего Stage 1 и
отдельно отражаться в итоговом решении.

## 6. Evaluation protocol

### Level 1 — частая validation

Фиксированный сбалансированный validation subset используется для регулярных
оценок, checkpoint selection и раннего обнаружения регрессий.

Логируются:

- WER и CER;
- validation loss;
- WER по доступным доменам;
- short-query WER;
- repetition и empty rates;
- скорость и память.

### Level 2 — полная validation

Перспективные checkpoints проверяются на полном validation split. Результат
включает общие и групповые метрики, длительности, короткие фразы и
катастрофические ошибки.

### Level 3 — закрытый test

Полный test запускается только после выбора checkpoint. Результаты test не
используются для изменения hyperparameters или повторного выбора checkpoint.

### Demo stress set

Дополнительный фиксированный набор проверяет неизвестные голоса, короткие
вопросы, разные акценты и реальные микрофоны. Он служит практическим gate, но
не заменяет официальный validation/test и не участвует в обучении.

## 7. Выбор checkpoint

Основная метрика выбора — macro-domain WER: WER считается отдельно по
доступным доменам или источникам, после чего значения усредняются между
группами. Это не позволяет крупному или простому домену скрыть провал другой
группы.

Отдельно сохраняются:

- best macro-domain WER;
- best overall WER;
- best short-query WER;
- best CER;
- best validation loss;
- periodic checkpoints;
- last checkpoint.

Основным кандидатом становится best macro-domain WER, если его общий WER не
показывает существенной регрессии относительно best overall WER.

## 8. Инициализация обучения

Проводится matched smoke двух вариантов:

| Run | Инициализация | Назначение |
|---|---|---|
| A | текущий лучший Stage 1 checkpoint | проверить быструю domain adaptation |
| B | AetherSpeech с нуля | проверить влияние старого domain bias |

Оба запуска используют одинаковые:

- train/validation data;
- порядок данных или воспроизводимый sampler;
- effective batch;
- бюджет audio hours и optimizer steps;
- evaluation protocol;
- scheduler family;
- seed, где это технически возможно.

Предварительный smoke budget — 10–20 тысяч optimizer steps. Точный бюджет
фиксируется после измерения examples, frames и audio hours на шаг.

Основной full run начинается с текущего checkpoint, если scratch не показывает
явного и устойчивого преимущества на неизвестных голосах, macro-domain WER и
catastrophic failure metrics.

Новый run всегда создаёт новый optimizer, scheduler, step counter и provenance.

## 9. Plateau и остановка

До запуска фиксируются:

- warmup boundary;
- evaluation interval;
- minimum WER improvement;
- plateau patience;
- hard ceiling;
- критерии аварийной остановки.

Предварительное правило plateau:

- warmup завершён;
- прошло минимум три eligible evaluation;
- macro-domain WER не улучшился минимум на 0.5 абсолютного процентного пункта;
- overall WER и short-query WER также не показывают устойчивого улучшения.

Если общий WER стоит, но худшие домены продолжают улучшаться, обучение не
останавливается автоматически только по общей метрике.

Прогресс логируется одновременно в:

- optimizer steps;
- audio hours seen;
- equivalent epochs;
- elapsed time и ETA.

Production training additionally has one persistent private W&B run URL. Its
identity is derived from the immutable local run ID, so checkpoint resume must
continue the same remote run. Successful W&B initialization is a preflight
requirement; after that point a monitoring-network failure must not stop local
training, checkpointing, or JSONL logging.

## 10. Диагностика обучения

Во время обучения и evaluation отслеживаются:

- train и validation loss;
- overall, macro-domain и per-domain WER/CER;
- short-query metrics;
- CTC blank rate;
- token entropy;
- повторяющиеся predictions;
- hypothesis/reference length ratio;
- gradient norms;
- activation statistics;
- learning rate;
- throughput;
- allocated и reserved VRAM;
- non-finite values;
- ошибки чтения данных.

## 11. Проверка пригодности для Stage 2

Хороший CTC результат не гарантирует улучшение speech-to-LLM интерфейса.
После выбора нового Stage 1 проводится matched transfer probe:

```text
frozen AetherSpeech
  → fresh small Connector
  → frozen Qwen
```

Старый и новый encoders сравниваются при одинаковых Connector architecture,
данных, Qwen, seed, budget и evaluation split.

Новый Stage 1 принимается как основа следующего Stage 2, если transfer probe
не хуже старого encoder и не показывает новых failure modes.

После принятия нового encoder все старые Stage 2 speech-state caches считаются
несовместимыми и пересобираются. Connector обучается заново.

## 12. Фазы работы

### Phase 0 — Dataset contract audit

Прочитать фактическую документацию и схему `manifestro/stage1_aether`,
зафиксировать immutable revision, splits, provenance, статистику и loader
contract. Ревизия `4bb733b62abd021c4a153ff5196682912933588e` зафиксирована;
manifest и validation schema проверены, loader и детерминированный
frame-budget sampler реализованы. Полная проверка checksum, uniqueness и
speaker overlap переносится в preflight на Pod.

### Phase 1 — Training/evaluation harness validation

Проверить forward, backward, gradient flow, targets, masks, decode, metrics,
checkpoint/resume, determinism, evaluation slices и аварийные guards.

### Phase 2 — Continuation vs scratch matched smoke

Сравнить две инициализации на одинаковом ограниченном бюджете.

### Phase 3 — Full Stage 1 training

Обучить выбранную инициализацию на полном train split до подтверждённого
plateau или hard ceiling.

### Phase 4 — Complete validation and closed test

Выбрать checkpoint по validation, выполнить полную validation и единственную
финальную test evaluation.

### Phase 5 — Voice and short-query stress evaluation

Проверить неизвестные голоса, акценты, короткие запросы, разные микрофоны и
repetition failures.

### Phase 6 — Matched Stage 2 transfer probe

Сравнить новый encoder со старым в одинаковом speech-to-LLM эксперименте.

### Phase 7 — Final decision

Принять или отклонить новый Stage 1 checkpoint по заранее определённым gates.

## 13. Definition of Done

Новая итерация Stage 1 считается завершённой, когда:

1. выполнен полный воспроизводимый training run;
2. выбран checkpoint только по validation;
3. full test WER ниже 15%;
4. macro-domain WER ниже 18%;
5. short-query WER ниже 15%;
6. repetition-collapse rate ниже 0.5%;
7. empty hypothesis rate ниже 0.5%;
8. отсутствует системный провал на неизвестных голосах;
9. catastrophic failure rate существенно ниже текущего Stage 1;
10. Stage 2 transfer probe не хуже старого encoder;
11. сохранены checkpoints, конфигурация, logs, provenance и итоговые метрики;
12. отдельно задокументированы ограничения и неподтверждённые свойства.

Stage 1 не заявляет универсальное понимание любого голоса. Он должен показать
измеримую устойчивость на разнообразных неизвестных голосах и устранить
наблюдавшийся класс критических ошибок на короткой разговорной речи.

## 14. За пределами текущего scope

В эту итерацию не входят без отдельного решения:

- обучение или fine-tuning Mimi;
- замена semantic codebook;
- одновременная смена архитектуры encoder и обучающих данных;
- внешний LM rescoring как основная метрика Stage 1;
- оптимизация Stage 2 Connector;
- Stage 3 и последующие продуктовые стадии;
- production deployment и публичная публикация весов.
