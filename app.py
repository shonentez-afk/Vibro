import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import random

st.set_page_config(page_title="🔧 Вибродиагностика", layout="wide", page_icon="🔧")

# ============================================================
# БАЗА ЗНАНИЙ ИЗ КНИГИ (12 разделов + 4 приложения)
# ============================================================

class KnowledgeBase:
    def __init__(self):
        self.chapters = {}
        self.checklists = {}
        self.glossary = {}
        self.bearing_tables = {}
        self.load_all()
    
    def load_all(self):
        # === РАЗДЕЛ 1: ТЕОРЕТИЧЕСКИЕ ОСНОВЫ ===
        self.chapters['1. Теоретические основы'] = """
## 🔬 Метод спектра огибающей (Envelope Analysis)

### Физика процесса
Дефект подшипника создаёт импульсы, модулирующие высокочастотные резонансы конструкции (2000–15000 Гц).

### Стадии развития дефекта:
| Стадия | Размер | Проявление | Время до отказа |
|--------|--------|------------|-----------------|
| 1 (начальная) | < 0,1 мм | Широкополосный шум | 6–12 мес. |
| 2 (развитая) | 0,1–1 мм | Пики BPFO/BPFI | 3–6 мес. |
| 3 (критическая) | 1–5 мм | Гармоники + боковые | 1–3 мес. |
| 4 (разрушение) | > 5 мм | Разрушение сепаратора | дни–часы |

### Алгоритм метода:
1. **Полосовая фильтрация** вокруг резонанса fc
2. **Детектирование огибающей** (преобразование Гильберта)
3. **БПФ огибающей** → спектр дефектных частот

### Правило 10×:
**fc ≥ 10 × F_вращ (Гц)**

💡 На СД-23 ищем резонансы в диапазоне 2000–8000 Гц.
"""

        # === РАЗДЕЛ 2: АППАРАТНОЕ ОБЕСПЕЧЕНИЕ ===
        self.chapters['2. Аппаратное обеспечение'] = """
## 🔧 Требования к датчикам для СД-23

| Параметр | Минимум | Оптимум |
|----------|---------|---------|
| Диапазон частот | 0,5 Гц – 10 кГц | 0,2 Гц – 15 кГц |
| Резонансная частота | > 20 кГц | > 25 кГц |
| Чувствительность | 10 мВ/g | 100 мВ/g |

### Рекомендуемые датчики:
- **АР-021** (ДИМЕКС, Россия)
- **АС-06** (ВАСТ, Россия)
- **352C33** (PCB, США) — эталон

### Способы установки:
| Способ | Верхняя частота | Применимость |
|--------|-----------------|--------------|
| Шпилька M5 | 25–30 кГц | ✅ Идеально |
| Клей | 15–20 кГц | ✅ Хорошо |
| Магнит | 5–8 кГц | ⚠️ Только fc ≤ 4000 Гц |
| Ручной щуп | 1–2 кГц | ❌ Непригодно |
"""

        # === РАЗДЕЛ 3: НАСТРОЙКА ПАРАМЕТРОВ ===
        self.chapters['3. Настройка параметров'] = """
## ⚙️ Выбор центральной частоты fc

| Обороты | fc, Гц | Применение |
|---------|--------|------------|
| < 1000 об/мин | 800 | Тихоходные машины |
| 1000–2000 | 1600 | Средние обороты |
| 1500 (стандарт) | **2500** | **Основной выбор** |
| > 2000 | 4000 | Высокие обороты |
| > 3000 | 5000–6300 | Шпиндели, ЧРП |

### Типы фильтров:
- **1/3 октавы** (СТАНДАРТ): f_low = fc/1.122, f_high = fc×1.122
- **1 октава**: f_low = fc/1.414, f_high = fc×1.414

### Разрешение спектра:
**Δf ≤ 0,5 Гц** — для боковых полос

### ⛔ ЗАПРЕЩЕНО для огибающей:
- ❌ Скорость (мм/с)
- ❌ Перемещение (мкм)

✅ **РАЗРЕШЕНО:**
- ✅ Ускорение (g peak)
- ✅ Ускорение (дБ)
"""

        # === РАЗДЕЛ 4: РАСЧЁТ ЧАСТОТ ===
        self.chapters['4. Расчёт частот дефектов'] = """
## 🧮 Основные формулы

### BPFO (наружное кольцо):
**BPFO = (n/2) × fr × (1 - d/D_p × cos α)**

### BPFI (внутреннее кольцо):
**BPFI = (n/2) × fr × (1 + d/D_p × cos α)**

### BSF (тело качения):
**BSF = (D_p/(2d)) × fr × (1 - (d/D_p × cos α)²)**

### FTF (сепаратор):
**FTF = (fr/2) × (1 - d/D_p × cos α)**

## ⚠️ КРИТИЧЕСКИ ВАЖНО:
**Дефект тела качения ищется на частоте 2×BSF, а НЕ BSF!**

### Проскальзывание:
Реальные частоты **ВСЕГДА НИЖЕ** расчётных на 1–3%.
"""

        # === РАЗДЕЛ 5: ИНТЕРПРЕТАЦИЯ ===
        self.chapters['5. Интерпретация спектров'] = """
## 🔍 Идентификация дефектов

### BPFO — наружное кольцо
- ✅ Пики на BPFO + гармоники (2×, 3×, 4×)
- ✅ **ОТСУТСТВИЕ боковых полос** (кольцо неподвижно)

### BPFI — внутреннее кольцо
- ✅ Пик на BPFI
- ✅ **БОКОВЫЕ ПОЛОСЫ с шагом 1×F_вращ** (ГЛАВНЫЙ ПРИЗНАК!)

### BSF — тело качения (2×BSF)
- ✅ Пик на **2×BSF** (НЕ BSF!)
- ✅ Боковые полосы с шагом **2×FTF**

### FTF — сепаратор
- ✅ Очень низкая частота (~0,4×F_вращ)
- 🚨 **АВАРИЙНАЯ СИТУАЦИЯ!**

## ⚠️ КАВИТАЦИЯ ИМИТИРУЕТ ДЕФЕКТ!
Если пики размытые — проверяйте кавитацию!
"""

        # === РАЗДЕЛ 6: КЕЙСЫ ===
        self.chapters['6. Практические кейсы (20 примеров)'] = """
## 📋 20 практических кейсов

### Кейс 1: Классический BPFO
**Объект:** Насос Grundfos CR-32, 1500 об/мин, 6309-2RS
**Результат:** BPFO=194,9 Гц (0,28 g) + 3 гармоники, БЕЗ боковых
**Диагноз:** BPFO, стадия 2

### Кейс 2: BPFI с модуляцией
**Объект:** Вентилятор ВЦ 6-28, 3000 об/мин, 6211-2RS
**Результат:** BPFI=324,6 Гц + боковые ±50 Гц
**Диагноз:** BPFI, стадия 1–2

### Кейс 3: Разрушение сепаратора (FTF)
**Объект:** Редуктор РЦД-800, 60 об/мин, 22320
**Результат:** Пики на 0,39; 0,78; 1,17 Гц
**Диагноз:** FTF, стадия 3 — **НЕМЕДЛЕННАЯ ОСТАНОВКА!**

### Кейс 4: Ложная кавитация
**Объект:** Насос НДВ 200-160, 3000 об/мин
**Первичный диагноз:** BPFO (пики 153 Гц)
**Проверка:** Широкополосный подъём 2–8 кГц → КАВИТАЦИЯ!

### Кейс 5-20: (см. полную версию книги)
- Дефект зуба шестерни
- Влияние ЧРП
- Пропуск из-за неверной fc
- Успешное прогнозирование
- Подшипник скольжения
- Looseness
- Тихоходная мельница
- Высокоскоростной шпиндель
- Ременная передача
- Вертикальный насос
- Дефект смазки
- Фреттинг-коррозия
- Электрический дефект
- Множественные дефекты
- Ложный дисбаланс
- Комбинированный BPFO+BPFI
"""

        # === РАЗДЕЛ 7: СПЕЦИФИЧЕСКИЕ СЛУЧАИ ===
        self.chapters['7. Специфические случаи'] = """
## ⚡ Специфические случаи

### Тихоходные машины (< 100 об/мин)
- fc = 800–1600 Гц
- Nlines = 6400–12800 (Δf ≤ 0,03 Гц)
- Усреднение 16–32 кадра

### Высокоскоростные шпиндели (> 10000 об/мин)
- fc = 6300 Гц
- Fmax = 5000–10000 Гц
- Установка **ТОЛЬКО на шпильку**

### Подшипники скольжения
⛔ **Метод огибающей НЕ ПРИМЕНИМ!**

### Кавитация
**Признаки:**
- Широкополосный подъём 2–8 кГц
- Нестабильные, размытые пики

### Помехи от ЧРП
**Правило:** fc ≥ f_carrier + 2×F_вращ_max
"""

        # === РАЗДЕЛ 8: ТРЕНДИНГ ===
        self.chapters['8. Трендинг и прогнозирование'] = """
## 📈 Трендинг

### Общий уровень огибающей:
- Норма: < 0,05 g
- Warning: 0,05–0,15 g
- Alarm: 0,15–0,40 g
- Critical: > 0,40 g

### Прогноз остаточного ресурса:
**Линейная модель:** t = (A_critical - A_current) / v
**Экспоненциальная модель:** t = ln(A_critical / A_current) / k
"""

        # === РАЗДЕЛ 9: СТАНДАРТЫ ===
        self.chapters['9. Стандарты и нормативы'] = """
## 📜 Стандарты

- **ISO 13373-2:2016** — Обработка и анализ виброданных
- **ISO 20816** — Оценка вибрации машин (зоны A-D)
- **ISO 15243** — Классификация повреждений подшипников
- **ГОСТ Р ИСО 13373-2-2016** — Российский аналог ISO 13373-2

### Зоны состояния по ISO 20816-3:
| Зона | Скорость, мм/с | Описание |
|------|----------------|----------|
| A | 0 – 1,8 | Новая машина |
| B | 1,8 – 4,5 | Длительная работа допустима |
| C | 4,5 – 11,2 | Нежелательна |
| D | > 11,2 | Повреждение возможно |
"""

        # === РАЗДЕЛ 10: ПО ===
        self.chapters['10. Программное обеспечение'] = """
## 💻 DREAM32 v.5 (ВАСТ)

**Основной инструмент для СД-23**

**Возможности:**
- ✅ Маршрутные измерения
- ✅ База ~5000 подшипников
- ✅ Автоматический расчёт частот
- ✅ Автоматическая диагностика
- ✅ Тренд-анализ
- ✅ Интеграция с 1С:ТОИР

### Сравнение ПО:
| Параметр | DREAM32 | @ptitude | CSI |
|----------|---------|----------|-----|
| СД-23 | ✅ | ❌ | ❌ |
| Русский | ✅ | ❌ | ❌ |
| Поддержка РФ | ✅ | ⚠️ | ❌ |
"""

        # === РАЗДЕЛ 11: ОШИБКИ ===
        self.chapters['11. Типичные ошибки'] = """
## ⚠️ 7 типичных ошибок

### Ошибка 11.1: Неверный выбор fc (25% случаев)
**Решение:** Широкополосный спектр → найти f_n

### Ошибка 11.2: Недостаточное разрешение (18%)
**Решение:** Nlines ≥ 3200, Δf ≤ 0,5 × F_вращ

### Ошибка 11.3: Игнорирование боковых полос (15%)
**Решение:** ВСЕГДА анализировать в дБ!

### Ошибка 11.4: Ложная кавитация (12%)
**Признаки кавитации:**
- Широкополосный подъём 2–8 кГц
- Размытые, нестабильные пики

### Ошибка 11.5: Пропуск раннего дефекта (15%)
**Решение:** Тренд + контрольные карты Шухарта

### Ошибка 11.6: Неверная интерпретация гармоник (8%)
**Правило:** Не ставить диагноз только по числу гармоник!

### Ошибка 11.7: Ошибки установки датчика (7%)
**Чек-лист:**
- ✅ Зачистка до металла
- ✅ Расстояние ≤ 150 мм
- ✅ Правильный способ крепления
"""

        # === РАЗДЕЛ 12: РЕСУРСЫ ===
        self.chapters['12. Инструменты и ресурсы'] = """
## 🛠 Онлайн-калькуляторы

| Калькулятор | База | Русский | Бесплатно |
|-------------|------|---------|-----------|
| vibro-expert.ru | ~2000 | ✅ | ✅ |
| SKF Calculator | >10000 | ❌ | ✅ |
| DREAM32 | ~5000 | ✅ | ✅ |

## Обучающие курсы

### Российские:
- **Учебный центр ВАСТ** — от 25 000 ₽
- **СевЗапУчЦентр** — от 30 000 ₽

### Международные:
- **Mobius Institute** — от 2500 USD
- **Vibration Institute** — от 2000 USD

## Форумы экспертов

- 🇷🇺 **vibro-expert.ru** — главный российский форум
- 🌍 **Eng-Tips Vibration Forum** — 15000 пользователей

## Литература

### На русском:
1. **Кузьмин А.В.** "Практика вибродиагностики" ⭐⭐⭐⭐⭐
2. **Барков А.В.** "Вибродиагностика подшипников качения" ⭐⭐⭐⭐⭐
3. **Кондратьев Е.Н.** "Вибродиагностика и вибромониторинг" ⭐⭐⭐

### На английском:
4. **Randall R.B.** "Vibration-Based Condition Monitoring" ⭐⭐⭐⭐⭐
5. **Harris T.A.** "Rolling Bearing Analysis" (библия) ⭐⭐⭐⭐⭐
"""

        # === ПРИЛОЖЕНИЕ А: ТАБЛИЦЫ ЧАСТОТ ===
        self.chapters['📎 Приложение А: Таблицы частот'] = """
## Таблицы частот для популярных подшипников

### 6208 (при 1500 об/мин):
- BPFO: 59,5 Гц
- BPFI: 90,5 Гц
- 2×BSF: 77,2 Гц
- FTF: 6,6 Гц

### 6308 (при 1500 об/мин):
- BPFO: 86,2 Гц
- BPFI: 138,8 Гц
- 2×BSF: 101,4 Гц
- FTF: 9,6 Гц

### 6309 (при 1500 об/мин):
- BPFO: 97,5 Гц
- BPFI: 154,5 Гц
- 2×BSF: 108,0 Гц
- FTF: 9,8 Гц

### 22210 (при 1500 об/мин):
- BPFO: 182,0 Гц
- BPFI: 318,0 Гц
- 2×BSF: 85,2 Гц
- FTF: 9,1 Гц

### Пересчёт при других оборотах:
**F_actual = F_table × (n_actual / n_table)**
"""

        # === ПРИЛОЖЕНИЕ Б: ШАБЛОНЫ ОТЧЁТОВ ===
        self.chapters['📎 Приложение Б: Шаблоны отчётов'] = """
## Шаблоны отчётов

### Краткий отчёт (1 страница):
- Объект, дата, инженер
- Диагноз
- Рекомендация
- Срочность

### Стандартный отчёт (3–5 страниц):
- Параметры измерения
- Результаты (таблица)
- Спектры огибающей
- Тренды
- Диагноз с обоснованием
- Рекомендации

### Полный отчёт по ISO 13373-2 (10+ страниц):
- Все данные стандартного отчёта
- Параметры измерения (fc, фильтр, разрешение)
- Расчёт дефектных частот
- Анализ боковых полос
- Фотографии дефектов
- Прогноз остаточного ресурса

### Экспертный отчёт (для руководства):
- Executive Summary
- Технический анализ
- Сравнение сценариев
- План действий
"""

        # === ПРИЛОЖЕНИЕ В: ЧЕК-ЛИСТЫ ===
        self.chapters['📎 Приложение В: Чек-листы'] = """
## Чек-листы для полевого инженера

### В.1. Подготовка к выходу:
- [ ] Заряд батареи СД-23 > 50%
- [ ] Маршрут загружен
- [ ] Датчик исправен
- [ ] СИЗ готовы

### В.2. Проведение измерения:
- [ ] Зачистка до металла
- [ ] Расстояние ≤ 150 мм
- [ ] Правильный способ крепления
- [ ] SNR > 10 дБ

### В.3. Анализ данных:
- [ ] Анализ в дБ-шкале
- [ ] Поиск боковых полос
- [ ] Расчёт K_shape
- [ ] Тренд-анализ

### В.4. Самопроверка ошибок:
- [ ] 11.1: Проверил выбор fc?
- [ ] 11.2: Δf ≤ 0,5 × F_вращ?
- [ ] 11.3: Проверил боковые в дБ?
- [ ] 11.4: Исключил кавитацию?
"""

        # === ПРИЛОЖЕНИЕ Д: СЛОВАРЬ ===
        self.chapters['📎 Приложение Д: Словарь терминов'] = """
## Словарь терминов (русско-английский)

### Основные термины:
- **Огибающая** — Envelope
- **Демодуляция** — Demodulation
- **Боковая полоса** — Sideband
- **Дефектная частота** — Fault Frequency
- **Полосовая фильтрация** — Bandpass Filtering
- **Усреднение** — Averaging
- **Проскальзывание** — Slippage

### Дефектные частоты:
- **BPFO** — Ball Pass Frequency, Outer race
- **BPFI** — Ball Pass Frequency, Inner race
- **BSF** — Ball Spin Frequency
- **FTF** — Fundamental Train Frequency

### Сокращения:
- **АЦП** — ADC (Analog-to-Digital Converter)
- **БПФ** — FFT (Fast Fourier Transform)
- **СКЗ** — RMS (Root Mean Square)
- **ЧРП** — VFD (Variable Frequency Drive)
- **ШИМ** — PWM (Pulse Width Modulation)
"""

        # === ЧЕК-ЛИСТЫ (интерактивные) ===
        self.checklists = {
            'В.1. Подготовка к выходу': [
                '✅ Заряд батареи СД-23 > 50%',
                '✅ Память свободна > 50%',
                '✅ Дата/время корректны',
                '✅ Калибровочный файл загружен',
                '✅ Проверка связи с ПК',
                '✅ Датчик ICP исправен',
                '✅ Маршрут загружен в СД-23',
                '✅ Инструктаж по ТБ пройден',
                '✅ СИЗ готовы'
            ],
            'В.2. Проведение измерения': [
                '✅ Установившийся режим',
                '✅ Обороты зафиксированы',
                '✅ Зачистка до металла (Ø30 мм)',
                '✅ Расстояние ≤ 150 мм',
                '✅ Нагрузочная зона',
                '✅ Правильный способ крепления',
                '✅ Уровень в норме',
                '✅ Нет клиппинга',
                '✅ SNR > 10 дБ'
            ],
            'В.3. Анализ данных': [
                '✅ Все данные загружены',
                '✅ Анализ в линейной шкале (g)',
                '✅ Анализ в логарифмической (дБ)',
                '✅ Поиск пиков (±3%)',
                '✅ Поиск боковых полос',
                '✅ Подсчёт гармоник',
                '✅ Расчёт K_shape',
                '✅ Тренд-анализ',
                '✅ Проверка на ложные срабатывания'
            ],
            'В.4. Самопроверка ошибок': [
                '✅ 11.1: Проверил выбор fc?',
                '✅ 11.2: Δf ≤ 0,5 × F_вращ?',
                '✅ 11.3: Проверил боковые в дБ?',
                '✅ 11.4: Исключил кавитацию?',
                '✅ 11.5: Сравнил с трендом?',
                '✅ 11.6: Правильно интерпретировал гармоники?',
                '✅ 11.7: Датчик установлен правильно?'
            ]
        }

        # === СЛОВАРЬ ТЕРМИНОВ ===
        self.glossary = {
            'Огибающая': 'Envelope — медленно меняющаяся функция, модулирующая амплитуду ВЧ сигнала',
            'Демодуляция': 'Demodulation — процесс выделения огибающей',
            'Боковая полоса': 'Sideband — спектральная компонента, симметричная основной частоте',
            'BPFO': 'Ball Pass Frequency Outer — частота прохождения тел через наружное кольцо',
            'BPFI': 'Ball Pass Frequency Inner — частота прохождения тел через внутреннее кольцо',
            'BSF': 'Ball Spin Frequency — частота вращения тела качения',
            'FTF': 'Fundamental Train Frequency — частота вращения сепаратора',
            'Кавитация': 'Cavitation — схлопывание пузырьков пара в жидкости',
            'Фреттинг': 'Fretting — повреждение при длительном простое',
            'Проскальзывание': 'Slippage — разница между теоретической и реальной частотой (1–3%)',
            'ЧРП': 'Частотный преобразователь (VFD)',
            'ШИМ': 'Широтно-импульсная модуляция (PWM)',
        }
    
    def search(self, query):
        results = []
        query_lower = query.lower()
        for name, content in self.chapters.items():
            if query_lower in content.lower():
                results.append({'chapter': name, 'content': content[:500] + '...'})
        return results


# ============================================================
# AI АГЕНТЫ
# ============================================================

class AIDiagnosticAgent:
    def __init__(self, kb):
        self.kb = kb
    
    def analyze_spectrum(self, bearing, rpm, defect, stage, freqs):
        analysis = {
            'defect_type': defect,
            'frequency': freqs.get(defect, freqs.get('2xBSF', 0)),
            'key_features': [],
            'recommendations': [],
            'book_reference': '',
            'warnings': [],
            'stage_info': ''
        }
        
        stage_descriptions = {
            'BPFO': {1: 'Только основная частота', 2: '2-я гармоника', 3: '2-3 гармоники', 4: 'Много гармоник'},
            'BPFI': {1: 'Слабый пик', 2: 'Чёткий + боковые', 3: 'Сильный + много боковых', 4: 'КРИТИЧНО!'},
            'BSF': {1: 'Слабый 2×BSF', 2: 'Чёткий 2×BSF', 3: 'Сильный + боковые 2×FTF', 4: 'Широкий пик'},
            'FTF': {1: 'Слабый низкочастотный', 2: 'Чёткий FTF', 3: 'Гармоники', 4: 'Сильная вибрация'}
        }
        analysis['stage_info'] = stage_descriptions.get(defect, {}).get(stage, '')
        
        if defect == 'BPFO':
            analysis['key_features'] = [
                f"✓ Пик на {freqs['BPFO']:.1f} Гц (BPFO)",
                f"✓ Гармоники: {freqs['BPFO']*2:.1f}, {freqs['BPFO']*3:.1f} Гц",
                "✓ НЕТ боковых полос (кольцо неподвижно)"
            ]
            analysis['recommendations'] = [
                "Проверить посадку наружного кольца",
                "Осмотреть на выкрашивание",
                "Запланировать замену через 3-6 мес"
            ]
            analysis['book_reference'] = "Раздел 5, Кейс 1"
            analysis['warnings'] = ["⚠️ Проверить кавитацию (раздел 7.5)"]
            
        elif defect == 'BPFI':
            analysis['key_features'] = [
                f"✓ Пик на {freqs['BPFI']:.1f} Гц (BPFI)",
                f"✓ Боковые полосы с шагом {freqs['fr']:.1f} Гц",
                "✓ Модуляция от вращения"
            ]
            analysis['recommendations'] = [
                "Проверить посадку на валу",
                "Осмотреть внутреннее кольцо",
                "Проверить биение вала"
            ]
            analysis['book_reference'] = "Раздел 5, Кейс 2"
            if stage == 4:
                analysis['warnings'] = ["🚨 КРИТИЧНО! НЕМЕДЛЕННАЯ ЗАМЕНА!"]
            else:
                analysis['warnings'] = ["⚠️ Боковые — главный признак BPFI"]
            
        elif defect == 'BSF':
            analysis['key_features'] = [
                f"✓ Пик на {freqs['2xBSF']:.1f} Гц (2×BSF!)",
                f"✓ Боковые с шагом {2*freqs['FTF']:.1f} Гц (2×FTF)",
                "✓ Нестабильная амплитуда"
            ]
            analysis['recommendations'] = [
                "Осмотреть тела качения",
                "Проверить сепаратор",
                "Проверить смазку"
            ]
            analysis['book_reference'] = "Раздел 5, Кейс 3"
            analysis['warnings'] = ["⚠️ Искать на 2×BSF, НЕ на BSF! (раздел 11.6)"]
            
        elif defect == 'FTF':
            analysis['key_features'] = [
                f"✓ Пик на {freqs['FTF']:.1f} Гц (FTF)",
                f"✓ Низкая частота (~{freqs['fr']*0.4:.1f} Гц)",
                "✓ Дефект сепаратора"
            ]
            analysis['recommendations'] = [
                "Проверить сепаратор",
                "Проверить зазоры",
                "Проверить смазку"
            ]
            analysis['book_reference'] = "Раздел 5, Кейс 3"
            analysis['warnings'] = ["🚨 АВАРИЙНАЯ СИТУАЦИЯ! Риск заклинивания!"]
        
        return analysis


class AITutorAgent:
    def __init__(self, kb):
        self.kb = kb
    
    def assess_user_level(self, correct, total):
        if total == 0: return 'Новичок 🌱'
        accuracy = correct / total
        if accuracy >= 0.9: return 'Эксперт 🔥'
        elif accuracy >= 0.7: return 'Продвинутый 📈'
        elif accuracy >= 0.5: return 'Средний 📚'
        return 'Новичок 🌱'
    
    def generate_personalized_tip(self, defect, level):
        tips = {
            'Новичок 🌱': {
                'BPFO': '📖 Кузьмин: BPFO = наружное кольцо, НЕТ боковых!',
                'BPFI': '📖 Кузьмин: BPFI = внутреннее + боковые с шагом fr',
                'BSF': '📖 Кузьмин: BSF ищи на 2×BSF! Частая ошибка',
                'FTF': '📖 Кузьмин: FTF ~ 0.4× оборотов'
            },
            'Средний 📚': {
                'BPFO': '📖 Барков: Смотри количество гармоник для стадии',
                'BPFI': '📖 Барков: Глубина модуляции = тяжесть',
                'BSF': '📖 Барков: Боковые 2×FTF — характерный признак',
                'FTF': '📖 Барков: Сравни с 0.4×fr'
            },
            'Продвинутый 📈': {
                'BPFO': '📖 Randall: Анализ соотношения амплитуд гармоник',
                'BPFI': '📖 Randall: Фазовые соотношения',
                'BSF': '📖 Randall: Влияние смазки на спектр',
                'FTF': '📖 Randall: Другие низкочастотные источники'
            },
            'Эксперт 🔥': {
                'BPFO': '📖 Harris: Комбинированные дефекты',
                'BPFI': '📖 Harris: Модели контакта',
                'BSF': '📖 Harris: Нестационарные сигналы',
                'FTF': '📖 Harris: Динамика сепаратора'
            }
        }
        return tips.get(level, tips['Новичок 🌱']).get(defect, '📖 Продолжайте обучение!')


# ============================================================
# БАЗА ПОДШИПНИКОВ
# ============================================================
BEARINGS_DB = {
    '6205':  {'n': 9,  'd': 7.94,  'D': 38.5,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6208':  {'n': 9,  'd': 12.70, 'D': 57.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6305':  {'n': 8,  'd': 11.00, 'D': 43.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6308':  {'n': 8,  'd': 15.08, 'D': 65.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6309':  {'n': 10, 'd': 15.875,'D': 72.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6310':  {'n': 8,  'd': 18.00, 'D': 80.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '6312':  {'n': 8,  'd': 22.00, 'D': 95.0,  'alpha': 0,  'type': 'Радиальный шариковый'},
    '22210': {'n': 20, 'd': 19.05, 'D': 70.0,  'alpha': 0,  'type': 'Сферический роликовый'},
    '22220': {'n': 32, 'd': 28.0,  'D': 135.0, 'alpha': 0,  'type': 'Сферический роликовый'},
    '32210': {'n': 17, 'd': 16.5,  'D': 68.0,  'alpha': 14, 'type': 'Конический роликовый'},
    '7014':  {'n': 24, 'd': 11.5,  'D': 95.0,  'alpha': 15, 'type': 'Угловой контактный'},
}


# ============================================================
# ФУНКЦИИ РАСЧЁТА И ГЕНЕРАЦИИ
# ============================================================

def calculate_frequencies(bearing_name, rpm):
    b = BEARINGS_DB[bearing_name]
    fr = rpm / 60.0
    d_D = b['d'] / b['D']
    cos_a = np.cos(np.radians(b['alpha']))
    return {
        'fr': fr,
        'BPFO': (b['n'] / 2) * fr * (1 - d_D * cos_a),
        'BPFI': (b['n'] / 2) * fr * (1 + d_D * cos_a),
        'BSF': (b['D'] / (2 * b['d'])) * fr * (1 - (d_D * cos_a)**2),
        'FTF': (fr / 2) * (1 - d_D * cos_a),
        '2xBSF': 2 * (b['D'] / (2 * b['d'])) * fr * (1 - (d_D * cos_a)**2)
    }

def generate_fault_signal(rpm, bearing_name, defect_type, stage, duration=0.3, fs=50000, resonance_freq=2500):
    t = np.arange(0, duration, 1/fs)
    freqs = calculate_frequencies(bearing_name, rpm)
    fr = freqs['fr']
    signal_data = np.random.normal(0, 0.02, len(t))
    
    if defect_type == 'none':
        return t, signal_data, freqs
    
    if defect_type == 'BPFO': fault_freq, mod_freq = freqs['BPFO'], None
    elif defect_type == 'BPFI': fault_freq, mod_freq = freqs['BPFI'], fr
    elif defect_type == 'BSF': fault_freq, mod_freq = freqs['2xBSF'], 2 * freqs['FTF']
    elif defect_type == 'FTF': fault_freq, mod_freq = freqs['FTF'], fr
    else: return t, signal_data, freqs
    
    amp_dict = {1: 0.15, 2: 0.4, 3: 0.8, 4: 1.5}
    amplitude = amp_dict.get(stage, 0.5)
    mod_dict = {1: 0.1, 2: 0.3, 3: 0.6, 4: 0.85}
    mod_depth = mod_dict.get(stage, 0.3) if mod_freq else 0
    harm_dict = {1: 1, 2: 2, 3: 3, 4: 5}
    n_harmonics = harm_dict.get(stage, 2)
    
    for h in range(1, n_harmonics + 1):
        current_freq = fault_freq * h
        current_amp = amplitude / h
        impulse_times = np.arange(0, duration, 1.0 / current_freq)
        for imp_time in impulse_times:
            modulation = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * imp_time)) if mod_freq else 1.0
            n_samples = int(0.003 * fs)
            impulse_t = np.arange(n_samples) / fs
            resonance = current_amp * modulation * np.sin(2 * np.pi * resonance_freq * impulse_t)
            decay = np.exp(-800 * impulse_t)
            start_idx = int(imp_time * fs)
            if start_idx + n_samples < len(signal_data):
                signal_data[start_idx:start_idx + n_samples] += resonance * decay
    return t, signal_data, freqs

def process_envelope(sig, fs, fc=2500, bandwidth='1/3 октавы'):
    if bandwidth == '1/3 октавы':
        f_low, f_high = fc / 1.122, fc * 1.122
    else:
        f_low, f_high = fc / 1.414, fc * 1.414
    if f_high >= fs / 2: f_high = (fs / 2) * 0.9
    sos = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    rectified = np.abs(filtered)
    sos_lp = signal.butter(4, min(1500, fs / 4), btype='low', fs=fs, output='sos')
    envelope = signal.sosfilt(sos_lp, rectified)
    N = len(envelope)
    yf = fft(envelope)
    xf = fftfreq(N, 1/fs)[:N//2]
    amplitude = 2.0/N * np.abs(yf[:N//2])
    mask = xf <= 1000
    return xf[mask], amplitude[mask], envelope


# ============================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================

if 'kb' not in st.session_state:
    st.session_state.kb = KnowledgeBase()
    st.session_state.ai_diag = AIDiagnosticAgent(st.session_state.kb)
    st.session_state.ai_tutor = AITutorAgent(st.session_state.kb)
    st.session_state.user_stats = {'correct': 0, 'total': 0}
    st.session_state.exam_question = None
    st.session_state.exam_show_answer = False


# ============================================================
# ИНТЕРФЕЙС
# ============================================================

st.title("🔧 AI Вибродиагностика PRO")
st.markdown("**Полное руководство: 12 разделов + 4 приложения + AI-агенты + 20 кейсов**")
st.markdown("**🌐 Развёрнуто на Streamlit Cloud | Автор: @tezzet**")

with st.sidebar:
    st.header("📚 Навигация")
    mode = st.radio("Выберите режим:", [
        "📊 Генератор спектров",
        "📖 Книга (12 разделов + 4 приложения)",
        "📎 Чек-листы",
        "📋 Стандарты ISO/ГОСТ",
        "🔍 Поиск по книге",
        "📖 Словарь терминов",
        "📝 AI Экзамен",
        "📈 Мой прогресс"
    ], index=0)
    
    st.divider()
    st.header("⚙️ Параметры")
    bearing = st.selectbox('🔩 Подшипник', list(BEARINGS_DB.keys()), index=3)
    st.caption(f"Тип: {BEARINGS_DB[bearing]['type']}")
    rpm = st.slider('🔄 Обороты', 300, 3600, 1500, 50)
    fc = st.selectbox('📡 fc (Гц)', [800, 1600, 2500, 4000, 5000, 6300], index=2)
    bandwidth = st.selectbox('🔧 Фильтр', ['1/3 октавы', '1 октава'], index=0)
    defect = st.selectbox('⚠️ Дефект', ['none', 'BPFO', 'BPFI', 'BSF', 'FTF'], index=1)
    stage = st.slider('📈 Стадия', 1, 4, 2)


# ============================================================
# РЕЖИМ: ГЕНЕРАТОР СПЕКТРОВ
# ============================================================

if mode == "📊 Генератор спектров":
    st.subheader("📊 Генерация спектра с AI-диагностом")
    
    freqs = calculate_frequencies(bearing, rpm)
    t, sig, _ = generate_fault_signal(rpm, bearing, defect, stage, resonance_freq=fc)
    xf_env, amp_env, envelope = process_envelope(sig, fs=50000, fc=fc, bandwidth=bandwidth)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle(f'{bearing} | {rpm} об/мин | {defect} | Ст.{stage} | fc={fc} | {bandwidth}', fontsize=13, fontweight='bold')
        
        axes[0].plot(t * 1000, sig, linewidth=0.3, color='steelblue')
        axes[0].set_title('Исходный сигнал')
        axes[0].set_xlim(0, 50)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t * 1000, envelope, linewidth=0.5, color='red')
        axes[1].set_title(f'Огибающая (fc={fc} Гц, {bandwidth})')
        axes[1].set_xlim(0, 50)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(xf_env, amp_env, linewidth=0.8, color='darkgreen')
        axes[2].set_title('Спектр огибающей (линейный)')
        axes[2].set_xlim(0, 500)
        axes[2].grid(True, alpha=0.3)
        
        amp_db = 20 * np.log10(amp_env + 1e-10)
        axes[3].plot(xf_env, amp_db, linewidth=0.8, color='purple')
        axes[3].set_title('Спектр огибающей (логарифмический)')
        axes[3].set_xlabel('Частота, Гц')
        axes[3].set_xlim(0, 500)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        if defect != 'none':
            ai = st.session_state.ai_diag.analyze_spectrum(bearing, rpm, defect, stage, freqs)
            
            st.subheader("🤖 AI Диагност")
            st.success(f"**Дефект:** {ai['defect_type']}")
            st.info(f"**Частота:** {ai['frequency']:.2f} Гц")
            st.warning(f"**Стадия {stage}:** {ai['stage_info']}")
            
            st.markdown("**📋 Признаки:**")
            for f in ai['key_features']:
                st.markdown(f)
            
            st.markdown("**💡 Рекомендации:**")
            for r in ai['recommendations']:
                st.markdown(f"• {r}")
            
            if ai['warnings']:
                st.markdown("**⚠️ Предупреждения:**")
                for w in ai['warnings']:
                    st.error(w)
            
            st.divider()
            st.subheader("🎓 AI Наставник")
            level = st.session_state.ai_tutor.assess_user_level(
                st.session_state.user_stats['correct'],
                st.session_state.user_stats['total']
            )
            tip = st.session_state.ai_tutor.generate_personalized_tip(defect, level)
            st.info(tip)
            st.caption(f"📖 {ai['book_reference']}")


# ============================================================
# РЕЖИМ: КНИГА
# ============================================================

elif mode == "📖 Книга (12 разделов + 4 приложения)":
    st.subheader("📖 Полное руководство по вибродиагностике")
    st.info("**12 разделов + 4 приложения • ~300 страниц • 20 практических кейсов • 7 типичных ошибок**")
    
    chapter = st.selectbox("Раздел:", list(st.session_state.kb.chapters.keys()))
    
    if chapter:
        st.markdown(st.session_state.kb.chapters[chapter])


# ============================================================
# РЕЖИМ: ЧЕК-ЛИСТЫ
# ============================================================

elif mode == "📎 Чек-листы":
    st.subheader("📎 Чек-листы для полевого инженера")
    
    checklist_name = st.selectbox("Чек-лист:", list(st.session_state.kb.checklists.keys()))
    
    if checklist_name:
        items = st.session_state.kb.checklists[checklist_name]
        
        st.markdown(f"**Всего пунктов:** {len(items)}")
        
        checked = {}
        for i, item in enumerate(items):
            checked[i] = st.checkbox(item, key=f"cl_{checklist_name}_{i}")
        
        st.divider()
        score = sum(1 for v in checked.values() if v)
        total = len(checked)
        progress = score / total if total > 0 else 0
        
        st.progress(progress)
        st.markdown(f"**Выполнено:** {score} из {total} ({progress*100:.0f}%)")
        
        if checklist_name == 'В.4. Самопроверка ошибок':
            if score == total:
                st.success("✅ ДИАГНОЗ ДОСТОВЕРЕН! Все пункты выполнены.")
            elif score >= total * 0.7:
                st.warning(f"⚠️ Проверка частично пройдена ({score}/{total}). Будьте осторожны.")
            else:
                st.error(f"❌ Диагностика недостоверна ({score}/{total}). Повторите измерения.")


# ============================================================
# РЕЖИМ: СТАНДАРТЫ
# ============================================================

elif mode == "📋 Стандарты ISO/ГОСТ":
    st.subheader("📋 Стандарты и нормативы")
    
    st.markdown("""
## Основные стандарты:

- **ISO 13373-2:2016** — Обработка и анализ виброданных
- **ISO 20816** — Оценка вибрации машин (зоны A-D)
- **ISO 15243** — Классификация повреждений подшипников
- **ГОСТ Р ИСО 13373-2-2016** — Российский аналог ISO 13373-2

## Пороговые значения (ISO 13373-2):

| Стадия | BPFO/BPFI, g | BSF, g | FTF, g |
|--------|--------------|--------|--------|
| 🟢 Внимание | 0.05-0.15 | 0.03-0.10 | 0.02-0.05 |
| 🟡 Тревога | 0.15-0.30 | 0.10-0.25 | 0.05-0.15 |
| 🔴 Критическая | > 0.30 | > 0.25 | > 0.15 |
    """)


# ============================================================
# РЕЖИМ: ПОИСК ПО КНИГЕ
# ============================================================

elif mode == "🔍 Поиск по книге":
    st.subheader("🔍 Поиск по всей книге")
    
    query = st.text_input("Введите запрос:", placeholder="например: кавитация, боковые полосы, fc")
    
    if query:
        results = st.session_state.kb.search(query)
        
        if results:
            st.success(f"Найдено совпадений: {len(results)}")
            for r in results:
                with st.expander(f"📖 {r['chapter']}"):
                    st.markdown(r['content'])
        else:
            st.warning("Ничего не найдено. Попробуйте другой запрос.")


# ============================================================
# РЕЖИМ: СЛОВАРЬ
# ============================================================

elif mode == "📖 Словарь терминов":
    st.subheader("📖 Словарь терминов (русско-английский)")
    
    for term, definition in st.session_state.kb.glossary.items():
        with st.expander(f"**{term}**"):
            st.markdown(definition)


# ============================================================
# РЕЖИМ: AI ЭКЗАМЕН
# ============================================================

elif mode == "📝 AI Экзамен":
    st.subheader("📝 AI Экзамен (на основе 20 кейсов)")
    
    level = st.session_state.ai_tutor.assess_user_level(
        st.session_state.user_stats['correct'],
        st.session_state.user_stats['total']
    )
    st.info(f"**Уровень:** {level}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Всего", st.session_state.user_stats['total'])
    with col2:
        st.metric("Правильных", st.session_state.user_stats['correct'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎲 Новый вопрос", use_container_width=True):
            if level in ['Новичок 🌱']:
                defects = ['BPFO', 'BPFI']
            elif level == 'Средний 📚':
                defects = ['BPFO', 'BPFI', 'BSF']
            else:
                defects = ['BPFO', 'BPFI', 'BSF', 'FTF']
            
            st.session_state.exam_question = {
                'bearing': random.choice(list(BEARINGS_DB.keys())),
                'rpm': random.choice([750, 1000, 1500, 2000, 3000]),
                'defect': random.choice(defects),
                'stage': random.randint(1, 4),
                'fc': random.choice([1600, 2500, 4000])
            }
            st.session_state.exam_show_answer = False
            st.rerun()
    
    with col2:
        if st.session_state.exam_question and not st.session_state.exam_show_answer:
            if st.button("👁️ Показать ответ", use_container_width=True):
                st.session_state.exam_show_answer = True
                st.session_state.user_stats['total'] += 1
                st.session_state.user_stats['correct'] += 1
                st.rerun()
    
    if st.session_state.exam_question:
        q = st.session_state.exam_question
        st.divider()
        
        st.markdown(f"""
### 📋 Параметры кейса
- **Подшипник:** {q['bearing']} ({BEARINGS_DB[q['bearing']]['type']})
- **Обороты:** {q['rpm']} об/мин
- **fc:** {q['fc']} Гц
- **Стадия:** {q['stage']}

**Задание:** Определите тип дефекта!
        """)
        
        freqs_ex = calculate_frequencies(q['bearing'], q['rpm'])
        t_ex, sig_ex, _ = generate_fault_signal(q['rpm'], q['bearing'], q['defect'], q['stage'], resonance_freq=q['fc'])
        xf_ex, amp_ex, _ = process_envelope(sig_ex, fs=50000, fc=q['fc'], bandwidth=bandwidth)
        
        fig_ex, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        ax1.plot(xf_ex, amp_ex, linewidth=0.8, color='darkgreen')
        ax1.set_title('Линейный спектр')
        ax1.set_xlim(0, 500)
        ax1.grid(True, alpha=0.3)
        
        amp_db_ex = 20 * np.log10(amp_ex + 1e-10)
        ax2.plot(xf_ex, amp_db_ex, linewidth=0.8, color='purple')
        ax2.set_title('Логарифмический спектр — ищите боковые полосы!')
        ax2.set_xlim(0, 500)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_ex)
        
        if st.session_state.exam_show_answer:
            ai_ex = st.session_state.ai_diag.analyze_spectrum(q['bearing'], q['rpm'], q['defect'], q['stage'], freqs_ex)
            st.success(f"**✅ Правильный ответ: {q['defect']}**")
            st.markdown("**Признаки:**")
            for f in ai_ex['key_features']:
                st.markdown(f)
            st.markdown(f"**Стадия {q['stage']}:** {ai_ex['stage_info']}")
            st.caption(f"📖 {ai_ex['book_reference']}")


# ============================================================
# РЕЖИМ: ПРОГРЕСС
# ============================================================

elif mode == "📈 Мой прогресс":
    st.subheader("📈 Ваш прогресс обучения")
    
    level = st.session_state.ai_tutor.assess_user_level(
        st.session_state.user_stats['correct'],
        st.session_state.user_stats['total']
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего вопросов", st.session_state.user_stats['total'])
    with col2:
        st.metric("Правильных", st.session_state.user_stats['correct'])
    with col3:
        acc = (st.session_state.user_stats['correct'] / max(1, st.session_state.user_stats['total']) * 100)
        st.metric("Точность", f"{acc:.1f}%")
    
    st.markdown(f"### 🏆 Ваш уровень: {level}")
    st.progress(acc / 100)
    
    st.divider()
    st.subheader("💡 Рекомендации AI")
    if acc < 50:
        st.warning("1. 📚 Начните с раздела 'Книга'\n2. 📕 Кузьмин А.В. 'Практика вибродиагностики'\n3. 🎓 Курс ВАСТ 'Основы вибродиагностики'\n4. 💬 Регистрация на vibro-expert.ru")
    elif acc < 70:
        st.info("1. 📕 Барков А.В. 'Вибродиагностика подшипников'\n2. ⚠️ Изучите раздел '7 типичных ошибок'\n3. 🔍 Практикуйте кейсы из раздела 6\n4. 📋 Используйте чек-листы")
    else:
        st.success("1. 📕 Harris T.A. 'Rolling Bearing Analysis' (библия)\n2. 🎓 Сертификация ISO 18436-2\n3. 👨‍🏫 Обучайте других\n4. 🏆 Mobius Institute Category III")


st.divider()
st.caption("© AI Вибродиагностика PRO | 12 разделов + 4 приложения + AI-агенты | ISO 13373-2 | СД-23 + DREAM32 | Автор: @tezzet")
