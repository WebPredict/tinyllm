# Domain Plan: Pulp Fiction & Weird Horror

A domain instance focused on early-to-mid 20th century pulp fiction and weird horror — Lovecraft, Howard, Chandler, Hammett, and their contemporaries. Public domain, distinctive style, entertaining output.

**Status:** Planning only — build after React/TS domain is proven.

---

## Why This Domain

- **Distinctive voice**: pulp fiction has a recognizable style that small models can mimic
- **Public domain**: most pre-1928 work (and many works through ~1950s) are freely available
- **No correctness needed**: unlike code, fiction just needs to feel right — no compiler
- **Great demo**: "my $60 model writes Lovecraftian horror" gets attention
- **Short form works**: pulp stories are 5K-20K words — fit well in training context
- **Strong patterns**: genre conventions, character archetypes, atmospheric descriptions

---

## Scope

### Primary genres (pre-1960, mostly public domain)

**Weird horror / cosmic horror**
- H.P. Lovecraft (~70 stories, all public domain)
- Clark Ashton Smith (short fiction, PD)
- Robert W. Chambers (The King in Yellow, PD)
- Algernon Blackwood (horror stories, PD)
- M.R. James (ghost stories, PD)
- Arthur Machen (supernatural fiction, PD)
- Lord Dunsany (fantasy/horror, PD)
- William Hope Hodgson (cosmic horror, PD)
- Ambrose Bierce (horror/supernatural, PD)

**Hardboiled detective / noir**
- Raymond Chandler (some stories in PD, novels entering PD)
- Dashiell Hammett (early stories PD)
- Carroll John Daly (Race Williams stories, PD)
- Erle Stanley Gardner (early pulp stories, PD)

**Adventure / sword and sorcery**
- Robert E. Howard (Conan, Solomon Kane — most PD)
- Edgar Rice Burroughs (Tarzan, John Carter — entering PD)
- H. Rider Haggard (King Solomon's Mines etc., PD)

**Science fiction pulp**
- H.G. Wells (PD)
- Jules Verne (PD)
- E.E. "Doc" Smith (Skylark series — some PD)
- Early Amazing Stories / Astounding Stories content

**General pulp / adventure**
- Sax Rohmer (Fu Manchu — PD, though problematic)
- Johnston McCulley (Zorro — PD)
- Lester Dent (Doc Savage — some PD)

### Style characteristics the model should learn

```
Pulp hallmarks:
  - Short, punchy sentences alternating with purple prose
  - Strong atmospheric openings
  - Vivid sensory descriptions
  - Archetype characters (hard-bitten detective, eldritch entity, etc.)
  - Cliffhanger chapter endings
  - First-person narration common
  - Period-appropriate vocabulary

Lovecraft specifically:
  - Baroque vocabulary (eldritch, cyclopean, gibbous, squamous)
  - Unreliable narrator discovering cosmic truths
  - Buildup through documents, letters, research
  - Climactic reveal of something indescribable
  - New England settings
  - Academic/antiquarian protagonists
```

---

## Data Sources

### Project Gutenberg (primary — free, bulk download)

```
Source:     gutenberg.org / gutenberg.org/cache/epub/
Method:     Bulk download + filter by author/subject
Volume:     Hundreds of books from target authors
Quality:    High (professionally scanned, proofread)
Format:     Plain text
```

Key authors on Gutenberg:
- Lovecraft: ~30+ works
- Poe (predecessor): ~70 works
- M.R. James: multiple collections
- Algernon Blackwood: multiple collections
- Arthur Machen: multiple works
- Ambrose Bierce: extensive works
- H.G. Wells: dozens of novels/stories
- Robert E. Howard: entering PD
- Edgar Rice Burroughs: many works now PD
- Conan Doyle: all Sherlock Holmes

### Archive.org (secondary — scanned pulp magazines)

```
Source:     archive.org (search "Weird Tales", "Black Mask", "Amazing Stories")
Method:     OCR'd text from scanned magazines
Volume:     Thousands of issues
Quality:    Medium (OCR errors, needs cleaning)
Format:     Various (PDF, text, DJVU)
```

Key magazines:
- **Weird Tales** (1923-1954): Lovecraft, Howard, Smith all published here
- **Black Mask** (1920-1951): Chandler, Hammett, hardboiled detective fiction
- **Amazing Stories** (1926+): early sci-fi pulp
- **Astounding Stories**: golden age sci-fi
- **Adventure**: general pulp fiction
- **Argosy**: all genres

### Fan-curated collections

- HP Lovecraft Archive (hplovecraft.com) — complete texts
- Robert E. Howard Foundation — texts entering PD
- Standard Ebooks (standardebooks.org) — beautifully cleaned PD texts
- Feedbooks PD catalog

### Estimated corpus size

```
Lovecraft complete works:        ~1.5MB
Other weird horror authors:      ~10-15MB
Hardboiled/noir:                 ~5-10MB
Adventure/sword & sorcery:       ~10-15MB
Sci-fi pulp:                     ~10-15MB
Poe, Doyle, and predecessors:    ~15-20MB
General Gutenberg fiction:        ~50-100MB+

Conservative estimate:  50-80MB curated pulp fiction
Aggressive estimate:    200-500MB with broader Gutenberg fiction

BPE tokens (at ~4 chars/token):
  Conservative: ~12-20M tokens
  Aggressive:   ~50-125M tokens
```

---

## Domain Assessment

| Dimension | Rating | Notes |
|---|---|---|
| Data availability | Moderate-high | Lots of PD text, but needs curation |
| Validator strength | Weak | No compiler — style is subjective |
| Correctness checkable | No | Fiction quality is subjective |
| Recommended model | 1B | Enough for fluent prose in a narrow style |
| Recommended tier | $60-90 | Same as React/TS 1B run |
| Hybrid value | Moderate | RAG for style examples, memory for plot continuity |

---

## Hybrid Modules for Fiction

### RAG — style reference retrieval

```
Query: "describe an abandoned New England mansion"
RAG retrieves: passages from Lovecraft describing similar settings
Model generates: in that style, grounded by the examples
```

This is the key hybrid module for fiction — the model generates in the style of whatever passages RAG retrieves.

### Structured memory — plot/character continuity

```
Track across a generation session:
  - Character names introduced
  - Setting established
  - Plot points mentioned
  - Tone/mood of current scene
  - POV character
```

Prevents the model from forgetting character names or changing settings mid-story.

### Template constraints — story structure

```
Pulp story template:
  1. Atmospheric opening (1-2 paragraphs)
  2. Protagonist introduction
  3. Inciting incident
  4. Rising tension (2-3 scenes)
  5. Climactic revelation/confrontation
  6. Denouement (brief)
```

Guide generation to follow story structure rather than rambling.

### Style validator (heuristic)

```
Check:
  - Sentence length variation (pulp alternates short/long)
  - Vocabulary richness (should use period-appropriate words)
  - Paragraph length (pulp uses short paragraphs)
  - Dialogue formatting (proper quotation, attribution)
  - No modern anachronisms (no "smartphone", "internet", etc.)
```

Not a compiler, but catches obvious style breaks.

---

## Training Approach

### Data preparation

1. Download complete works from Gutenberg for target authors
2. Download cleaned texts from Standard Ebooks
3. Strip headers/footers/legal notices from Gutenberg texts
4. Quality filter: remove OCR artifacts, tables of contents, indices
5. Split into training chunks (~2K-4K tokens each)
6. Train BPE tokenizer (8K vocab — smaller vocab for literary text)

### Training mix

```
40%  Weird horror (Lovecraft, Smith, Blackwood, Machen, Chambers)
20%  Hardboiled detective (Chandler, Hammett, Daly)
20%  Adventure/sword & sorcery (Howard, Burroughs)
10%  Sci-fi pulp (Wells, Verne, early Amazing Stories)
10%  Gothic predecessors (Poe, Stoker, Shelley)
```

### Instruction tuning pairs (for later)

```
"Write a Lovecraftian opening paragraph about a fishing village"
"Continue this noir scene: The dame walked in like trouble on heels."
"Describe an ancient temple in Robert E. Howard's style"
"Write dialogue between a detective and a suspect"
```

---

## Demo Concepts

### "The Pulp Machine"

Interactive fiction generator with style controls:

```
Genre: [Horror / Detective / Adventure / Sci-fi]
Style: [Lovecraft / Chandler / Howard / Wells]
Prompt: "A strange letter arrives at Miskatonic University"

[Generate] → produces 500 words in the selected style
[Continue] → extends the story
[New scene] → starts a new scene in the same story
```

### "Style Transfer"

Take a mundane scene and rewrite it in different pulp styles:

```
Input: "A man walked into a bar and ordered a drink."

Lovecraft: "The gaunt figure shambled into that ill-regarded 
establishment on the waterfront, its shadow falling across 
the sawdust floor in a manner that seemed to defy the wan 
gaslight above..."

Chandler: "He came in from the rain like a wet dog looking 
for a warm spot. The bartender gave him the kind of look 
you give a stain on your best shirt."

Howard: "The massive Cimmerian shouldered through the 
tavern door, his blue eyes sweeping the smoke-filled 
room with the wariness of a jungle cat..."
```

### "Infinite Weird Tales"

Generate a full issue of a pulp magazine:

```
WEIRD TALES — Generated Issue
  
  "The Whisperer in the Walls" — a cosmic horror tale
  "Blood on the Nile" — a sword & sorcery adventure  
  "The Maltese Dagger" — a hardboiled mystery
  "The Last Martian" — a science fiction tale
```

---

## Estimated Effort and Cost

| Phase | Time | Cost |
|---|---|---|
| Data collection (Gutenberg + cleaning) | 1-2 days | $0 |
| Tokenizer training | Minutes | $0 |
| 1B model training (4x H100) | ~5-7 hours | ~$70-90 |
| Instruction tuning (LoRA) | 1 hour | $0 (laptop) |
| RAG index building | 30 min | $0 (laptop) |
| Style validator | Half day | $0 |
| Demo/interface | 1 day | $0 |
| **Total** | **~3-4 days** | **~$70-90** |

---

## When to Build

After the React/TS 1B model is working with hybrid modules. The infrastructure is identical — just swap the data, tokenizer, validators, and RAG corpus. The domain framework (Part XVII) should make this a quick spin-up.

This domain would demonstrate that the hybrid architecture works beyond code — proving the "any domain" claim from the plan.
