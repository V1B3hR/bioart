# 🛣️ Highway Map — bioart (krytyczna ścieżka)

Highway = minimalna, najszybsza i mierzalna ścieżka dowiezienia wartości od PoC do pierwszego publicznego wydania, z jasno zdefiniowanymi bramkami (Go/No‑Go).

Powiązane dokumenty:
- Vision/strategy: [roadmap.md](./roadmap.md)
- Execution details: [pathwaymap.md](./pathwaymap.md) (szczegółowe checklisty, SLO, DoD)

## 1) Cele „Highway” i ograniczenia

- Cel release R1:
  - Dostarczyć stabilny rdzeń (VM + I/O), podstawowy interfejs (CLI lub proste API), oraz jeden działający adapter „sandbox” do symulowanej/safe integracji.
- Ograniczenia (SLO/SLA minimalne):
  - Współczynnik sukcesu ≥ 99% dla ścieżki krytycznej
  - MTTR ≤ 30 min (na podstawie runbooków)
  - Koszt/100 zleceń ≤ ustalony próg (TBD w pathwaymap.md)
- Zakres R1 (co wchodzi):
  - Packaging (biblioteka + CLI), podstawowa dokumentacja i quickstart
  - Telemetria bazowa (logi strukturalne, metryki czasu i błędów)
  - Adapter „sandbox” + testy E2E
- Poza zakresem R1 (przesunięte):
  - Integracje produkcyjne z wet‑lab, pełny GUI/playground, akceleratory (FPGA), zaawansowane ECC/ML

## 2) Krytyczna ścieżka (kamienie milowe i zależności)

- M0: Hardening PoC i repo
  - Deliverables: format/lint/CI, test harness, walidacja konfiguracji, logi strukturalne
  - Zależności: brak
  - Go/No‑Go: „zielone” CI, ≥60% pokrycia w modułach kluczowych, reprodukowalne dev‑setup
- M1: Profilowanie i pierwsze optymalizacje
  - Deliverables: profil hot‑path, eliminacja top‑N wąskich gardeł, cache (z TTL)
  - Zależności: M0
  - Go/No‑Go: ≥30% skrócenia czasu ścieżki krytycznej
- M2: Telemetria kosztów i guardraile
  - Deliverables: instrumentacja kosztów/iterację, budżety i alerty anomalii
  - Zależności: M0
  - Go/No‑Go: raport kosztów/100 zleceń w dopuszczalnym progu
- M3: Porty/adaptery + „sandbox” E2E
  - Deliverables: interfejsy (porty), mock + adapter sandbox, testy kontraktów i E2E
  - Zależności: M0, M1 (stabilność), M2 (limity)
  - Go/No‑Go: pełny, audytowalny trace zlecenia w sandboxie
- M4: Niezawodność i odporność
  - Deliverables: retry/backoff + jitter, idempotencja, circuit‑breaker, limity równoległości
  - Zależności: M3
  - Go/No‑Go: testy awaryjne/chaos ≥95% pass
- M5: Dystrybucja (R1)
  - Deliverables: pakiet (PyPI lub inny), CLI, dokumentacja (README, Quickstart, Runbook), SBOM, release notes
  - Zależności: M3–M4
  - Go/No‑Go: instalacja „od zera” <15 min, smoke E2E zielone, SLO spełnione
- M6: Stabilizacja po release
  - Deliverables: triage feedbacku, bugfixy, uzupełnienie docs, metryki użycia
  - Zależności: M5
  - Go/No‑Go: brak blockerów P0/P1 przez 7 dni

Viz. zależności (D = dependency chain):
- D1: M0 → M1 → M3 → M4 → M5 → M6
- D2 (równoległe): M2 może biec po M0 i przed M3 (włącza limity i raporty)

## 3) Minimalny zakres PR‑ów na „Highway”

- PR‑01: Infrastruktura repo (format/lint/CI, CODEOWNERS, SECURITY, CONTRIBUTING)
- PR‑02: Test harness + pierwsze testy ścieżki krytycznej
- PR‑03: Logi strukturalne + metryki bazowe
- PR‑04: Profilowanie + optymalizacje hot‑path (1/2)
- PR‑05: Instrumentacja kosztów + budżety + alerty
- PR‑06: Porty/adaptery: interfejsy + mock + testy kontraktów
- PR‑07: Adapter „sandbox” + E2E + retry/backoff + limits
- PR‑08: Packaging + CLI + dokumentacja użytkownika + release pipeline
- PR‑09: Stabilizacja po R1 (bugfixy + feedback)

Zasady:
- Jedna zmiana logiczna na PR; testy i docs w tym samym PR.
- Rozmiar PR ukierunkowany na szybki review (<30 min).
- Każdy PR referencją do tracking issue „Highway R1”.

## 4) Kryteria jakości (skrót)

- Wydajność: skrócenie latencji ścieżki krytycznej o ≥30% vs baseline
- Niezawodność: sukces ≥99%, chaos tests ≥95% pass
- Koszt: koszt/100 zleceń ≤ próg (def. w pathwaymap.md)
- Obserwowalność: logi strukturalne + metryki czasu/błędów, podstawowy tracing
- Bezpieczeństwo: zarządzanie sekretami, least privilege, audit trail dla operacji zewnętrznych

## 5) Ryzyka o wysokim wpływie i mitigacje

- Eskalujący koszt przy wolumenie → cache/batching, budżety, alerty anomalii
- Niestabilność API zewnętrznych → adapter pattern, feature flags, circuit‑breaker, sandbox
- Nieprzewidywalne formaty danych → kontrakty i walidacja schematów, testy kontraktowe
- Rozjechane środowiska dev → kontenery/lockfile, pinned wersje, reproducible builds

## 6) Mechanika egzekucji

- Tracking: 1 epic/issue „Highway R1” + board (todo/in‑progress/review/done)
- Cadence: krótkie PR‑y, częste merge, release candidate przed R1
- Dokumentacja decyzji: ADR w `docs/adr/` dla kluczowych wyborów
- Runbooki: `docs/runbooks/` (incident response, release, rollback)

## 7) Linki i źródła

- Vision: [roadmap.md](./roadmap.md)
- Execution: [pathwaymap.md](./pathwaymap.md)

## 8) Historia zmian

- v0.1 — pierwsza wersja Highway Map (niniejsza)
