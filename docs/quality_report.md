# UltraChat-Sinhala — Translation Quality Report

_Generated 2026-06-26 04:30 · `tools/quality_report.py` (full-scan, deterministic)_

This report measures the **translated Sinhala output** of the cleaned dataset (post ZWJ repair) across structural, linguistic, orthographic, and duplication dimensions. All metrics are computed over every record (no sampling).

## Summary

| split | records | turns | JSON-invalid | untranslated turns | repetition turns | dup prompt_ids | recs w/ ZWJ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT (cleaned+masking-fixed) | 230,975 | 1,461,892 | 0.000% | 0.017% | 0.501% | 0.000% | 99.884% |
| GEN (cleaned+masking-fixed) | 284,336 | 1,510,092 | 0.000% | 0.016% | 0.473% | 0.000% | 99.599% |

## Methodology

- **Scope:** full scan of all shards; metrics computed on the translated text only (source-aligned length-ratio is noted as future work).
- **Language/script:** per message, the Sinhala-letter share = `Sinhala letters / (Sinhala + Latin letters)`, judged only when ≥20 letters are present. A *low-Sinhala* message scores <0.5; an *untranslated* message has 0 Sinhala letters and ≥20 Latin (NLLB left it in English). Some Latin is expected and legitimate — code, URLs, identifiers, brand names.
- **Repetition:** for messages ≥200 chars, the zlib compression ratio (compressed/raw bytes) is a degeneracy proxy; ratios below 0.18 indicate looping/repeated spans.
- **Orthography:** ZWJ (U+200D) counts confirm the conjunct repair landed; virama+space+consonant counts are reported for transparency but are dominated by legitimate word boundaries.
- **Duplication:** exact duplicate `prompt_id`s and exact duplicate concatenated content (md5).

## SFT (cleaned+masking-fixed)

- Shards analysed: **10**
- Records (dialogues): **230,975**  ·  messages (turns): **1,461,892**

### Structural integrity

| check | count | rate |
|---|---:|---:|
| JSON-invalid lines | 0 | 0.000% |
| Schema-invalid records | 0 | 0.000% |
| Empty turns | 37 | 0.003% |
| Role-alternation violations | 0 | 0.000% |

### Language / script (translated text)

| metric | value |
|---|---:|
| Mean Sinhala-letter share per message | 0.9733 |
| Median Sinhala-letter share | 1.0000 |
| 1st-percentile Sinhala share | 0.4594 |
| Low-Sinhala messages (<50% Sinhala letters) | 16,575 (1.134%) |
| Untranslated messages (0 Sinhala, ≥20 Latin) | 247 (0.017%) |

### Failure modes

| metric | value |
|---|---:|
| Empty turns | 37 (0.003%) |
| Degenerate-repetition messages (zlib ratio < 0.18, ≥200 chars) | 7,317 (0.501%) |
| Median compression ratio (long messages) | 0.307 |

### Orthography — ZWJ / conjuncts

| metric | value |
|---|---:|
| Total ZWJ (U+200D) joiners | 11,489,203 |
| Records containing ≥1 joiner | 230,707 (99.884%) |
| Mean joiners per record | 49.7 |
| Virama+space+consonant remaining¹ | 24,920,343 |

> ¹ Mostly *legitimate* word-final virama before the next word, not errors. The lexicon-gated repair already merged the ~1.3M attested conjuncts per shard; see the repair audit (`sft_zwj_audit.md`) for the small out-of-scope residual.

### Duplication

| check | count | rate |
|---|---:|---:|
| Duplicate prompt_ids | 0 | 0.000% |
| Duplicate content (md5 of turns) | 0 | 0.000% |

### Size distribution

| metric | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| Message chars | 447 | 1,978 | 4,268 | 22,649 |
| Record chars | 4,722 | 8,781 | 13,155 | 33,175 |
| Turns / record | 6 | 8 | 14 | 14 |

## GEN (cleaned+masking-fixed)

- Shards analysed: **10**
- Records (dialogues): **284,336**  ·  messages (turns): **1,510,092**

### Structural integrity

| check | count | rate |
|---|---:|---:|
| JSON-invalid lines | 0 | 0.000% |
| Schema-invalid records | 0 | 0.000% |
| Empty turns | 60 | 0.004% |
| Role-alternation violations | 0 | 0.000% |

### Language / script (translated text)

| metric | value |
|---|---:|
| Mean Sinhala-letter share per message | 0.9769 |
| Median Sinhala-letter share | 1.0000 |
| 1st-percentile Sinhala share | 0.5561 |
| Low-Sinhala messages (<50% Sinhala letters) | 12,157 (0.805%) |
| Untranslated messages (0 Sinhala, ≥20 Latin) | 247 (0.016%) |

### Failure modes

| metric | value |
|---|---:|
| Empty turns | 60 (0.004%) |
| Degenerate-repetition messages (zlib ratio < 0.18, ≥200 chars) | 7,148 (0.473%) |
| Median compression ratio (long messages) | 0.316 |

### Orthography — ZWJ / conjuncts

| metric | value |
|---|---:|
| Total ZWJ (U+200D) joiners | 10,670,352 |
| Records containing ≥1 joiner | 283,195 (99.599%) |
| Mean joiners per record | 37.5 |
| Virama+space+consonant remaining¹ | 22,986,607 |

> ¹ Mostly *legitimate* word-final virama before the next word, not errors. The lexicon-gated repair already merged the ~1.3M attested conjuncts per shard; see the repair audit (`sft_zwj_audit.md`) for the small out-of-scope residual.

### Duplication

| check | count | rate |
|---|---:|---:|
| Duplicate prompt_ids | 1 | 0.000% |
| Duplicate content (md5 of turns) | 0 | 0.000% |

### Size distribution

| metric | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| Message chars | 291 | 1,884 | 4,394 | 39,724 |
| Record chars | 3,445 | 6,921 | 10,482 | 43,707 |
| Turns / record | 5 | 7 | 13 | 13 |

## Flagged examples (qualitative)

### SFT (cleaned+masking-fixed)

**Untranslated (Latin-only) messages** (showing 6):

- `d45493fd8520…` — ```python def is_prime(number):     """     This function takes in a positive integer and checks whether it is prime or not.     It returns True if the number is a prime number, and False otherwise.  
- `0966411bbdcf…` — ```c++ #include <iostream> #include <unordered_set>  // Function to check if list has non-unique integers bool hasDuplicates(int arr[], int n) {     // Create unordered set to store unique integers   
- `6b5719a7b916…` — ``` using System;  namespace RandomNumberGenerator {     class Program     {         static void Main(string[] args)         {             int lowerLimit;             int upperLimit;             int r
- `ead001d3fc4a…` — ``` import java.util.Scanner;  public class CircleDiameter {    public static void main(String[] args) {              Scanner scanner = new Scanner(System.in);              System.out.print("Enter the
- `a87361f5a3d3…` — <Tag key="Name" value="webserver"/>
- `06bddeb13bdb…` — ``` # This program takes in a text file as input and outputs the frequency of each word in the file.  # Open and read the file file = File.read("input_file.txt")  # Remove any special characters, numb

**Low-Sinhala (<50%) messages** (showing 6):

- `4ce0e09b6eaa…` — ඔව්, මෙන්න අන්තර්ජාලයේ ඇති සම්පත් කිහිපයක් ඔබට විවිධ කවි වර්ග සහ ඒවා ලියන ආකාරය ගැන වැඩිදුර ඉගෙන ගත හැකිය:  1. සෞඛ්‍යය කවි පදනම (https://www.poetryfoundation.org/learn) 2. සෞඛ්‍යය ලේඛකයින්ගේ සාරාංශය (
- `d3e36a4af394…` — ඔව්, මෙන්න ඔයා විස්තර කරපු ක්‍රියාකාරීත්වය ලබාගන්න පුළුවන් උදාහරණ වැඩසටහනක්.  ```java import java.io.BufferedReader; import java.io.FileReader; import java.io.IOException; import java.util.*;  public 
- `d3e36a4af394…` — අනිවා, මට සතුටුයි කේතය ඔයාට හොදින් වැඩ කරනවා! ජාවා වලදී, `Map` අතුරු මුහුණත යතුරු අගයන් වෙත සිතියම්ගත කරන වස්තුවක් නියෝජනය කරයි. ඒකෙන් අගයන් එකතු කරන්න, ඉවත් කරන්න, සහ නැවත ලබාගන්න ක්‍රම ලබා දෙනවා. ජා
- `d3e36a4af394…` — අනිවා, මම සතුටින් පැහැදිලි කරන්නම්!  ජාවා වලදී, `HashSet` යනු අනුපිටපත් අගයන් ප්‍රතික්ෂේප කරමින් අද්විතීය මූලද්‍රව්‍ය ගබඩා කරන එකතුවකි. එය හෑෂ් වගුවක් භාවිතයෙන් ක්‍රියාත්මක වන අතර එය එකතු කිරීම, ඉවත් 
- `12aa9c782c8b…` — අනිවා, මෙන්න ඒ කේතයේම සංශෝධිත අනුවාදයක්:  ``` ;; Define a function named concatenate-and-remove-duplicates that takes two parameters: s1 and s2. (defn concatenate-and-remove-duplicates [s1 s2]   ;; Us
- `13b2324264d9…` — මට ඩිස්ක් එකට කෙලින්ම ෆයිල් ලියන්න බෑ. පහල තියෙන්නේ ශබ්දකෝෂ ලැයිස්තුවක් හදන්න කේතය, එහිදී සෑම ශබ්දකෝෂයක්ම නියෝජනය කරන්නේ පුද්ගලයෙකුගේ නම, වයස සහ වෘත්තිය සඳහා යතුරු සහිතව, පසුව එම දත්ත json ගොනුවකින් ක

**Degenerate-repetition messages** (showing 6):

- `281fabb7abf7…` — මේ වට්ටෝරුව නම් හරිම රසයි! සුප් එකට අතුරුපසක් හෝ ආධාරකයක් නිර්දේශ කරන්න පුළුවන්ද? ඒ වගේම, ඔයා මට වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන වෙන
- `29c1d8f1102e…` — අනිවා, මට පුළුවන් ගෙවීම් කාලසටහන ගැන විස්තරාත්මක විස්තරයක් දෙන්න. මෙන්න ඔයාට පුළුවන් කොහොමද ගණනය කරන්න එක් එක් ගෙවීමේ බිඳවැටීම:  1. සෞඛ්‍යය මාසික පොලී අනුපාතය ගණනය කරන්න:  මාසික පොලී අනුපාතය = වාර්ෂික
- `d04acfd713ae…` — හිටපු Mortlake ක්‍රිකට් ක්‍රීඩිකාවක වන Georgia Wareham ට අඟහරුවාදා රාත්‍රියේ ඇගේ ජීවිතය සදහටම වෙනස් කරන ඇමතුමක් ලැබුණා. ජාතික තේරීම් නිලධාරී ෂෝන් ෆ්ලෙග්ලර් දුරකථනයේ අනෙක් කෙළවරේ සිටි අතර වොරේහැම්ට දැන
- `76fb4db9e063…` — පෙරවදන  ජල අවශ්‍යතාවය වැඩිවීම සහ දේශගුණික විපර්යාසයන් ජල සැපයුමට ඇති කරන බලපෑම හේතුවෙන් ජල සංරක්ෂණය මෑත වසරවලදී තීරණාත්මක ගැටළුවක් බවට පත්ව ඇත. නිවාසවල තිරසාර ජල භාවිතය ප්‍රවර්ධනය කිරීම අරමුණු කරගත් ප
- `86e9d3ddef0c…` — ඔව්, මෙන්න එළවළු කෑම වේලක් ඔයාගේ අච්චාරු දාපු ෆිලෙට් මිග්නොන් සහ තම්බපු සුදුළූණු පිරූ අර්තාපල් එක්ක හොඳින් ගැලපෙනවා:  අර්තාපල් සමග අච්චාරු කළ කොළ බෝංචි  අමුද්‍රව්‍ය: - අළු පාට බෝංචි 1 lb, කපාගත් - පිට
- `19dad2154e89…` — දශම කාර්යය ද්විමය (දශම) { if (දශම අංකය === 0) { නැවත ලබාදෙන්නේ "0" ලෙසයි. { වෙන දෙයක් නම් (දශම අංක === 1) { නැවත "1" ලබා දෙන්න. වෙන දෙයක් දශම ප්‍රතිපල ද්විමය (Binary) ගණිතමය (මත) දශම ප්‍රතිපල (දශම ප්‍

**Empty turns** (showing 6):

- `9c9477b56d43…` — <empty turn>
- `8f8e0b1e2de9…` — <empty turn>
- `1c14d6012a77…` — <empty turn>
- `fd42655fbf61…` — <empty turn>
- `d798d243b239…` — <empty turn>
- `b694b4fc9ec6…` — <empty turn>

### GEN (cleaned+masking-fixed)

**Untranslated (Latin-only) messages** (showing 6):

- `9a630e6c9e5b…` — ``` #include <iostream> using namespace std;  // Node structure struct Node {     int data;     Node *next; };  // Linked list class class LinkedList { private:     Node *head;  public:     LinkedList
- `854b9f9415cf…` — ``` import 'dart:math';  void main() {   double sideA = 3.0;   double sideB = 4.0;    double hypotenuse = sqrt(sideA * sideA + sideB * sideB);    print("The length of the hypotenuse is $hypotenuse"); 
- `091c9cc53088…` — ```julia function matrix_transpose()     # Prompt user to enter matrix dimensions     print("Enter the number of rows: ")     m = parse(Int64, readline())      print("Enter the number of columns: ")  
- `41372649bce9…` — ``` struct Queue<T> {     data: Vec<T>, }  impl<T> Queue<T> {     fn new() -> Self {         Queue { data: Vec::new() }     }          fn enqueue(&mut self, item: T) {         self.data.push(item);   
- `321918fda7da…` — ``` # Program to calculate the determinant of a square matrix  function determinant(matrix::Array{Float64,2})     # Check if the matrix is square     if size(matrix,1) != size(matrix,2)         error(
- `640ba2492c48…` — ``` function arrayToString(arr, separator) {    return arr.join(separator); }  //example usage const arr = ['Hello', 'World!', 'How', 'are', 'you?']; const separator = " "; const result = arrayToStrin

**Low-Sinhala (<50%) messages** (showing 6):

- `db4928006d43…` — මට පරිශීලක අතුරු මුහුණතට හෝ ආදාන / ප්‍රතිදාන හැකියාවන්ට ප්‍රවේශ විය නොහැක. කෙසේ වෙතත්, පහත දැක්වෙන්නේ C++ හි පූර්ණ සංඛ්‍යා ලැයිස්තුවක මාදිලිය ගණනය කිරීම සඳහා වන කේතයයි:  ```c++ #include <iostream> #in
- `ff4ad1c05e46…` — අනිවාර්යෙන්ම! මෙන්න Tcl script එකක් අක්ෂර වින්‍යාසය අනුව string ලැයිස්තුවක් වර්ග කරනවා:  ``` set list {"apple" "banana" "cherry" "date" "elderberry"} set sorted_list [lsort $list] puts $sorted_list ``
- `7fc24d279e0b…` — මෙන්න C# වල අරා භාවිතා කරමින් ස්ටැක් එකක් ක්‍රියාත්මක කිරීමේ උදාහරණයක්:  ``` using System;  public class Stack {     private int[] data;     private int top;      public Stack(int size)     {         
- `4bb3e00423c0…` — අනිවා, අපිට පුළුවන් මේක වගේ පසුබිම් වර්ණයක් එකතු කරන්න:  ``` html {   background: #333333; /* fallback background color */      background: linear-gradient(to right, #333333, #ffffff); /* horizontal g
- `9db29ab97bdf…` — මෙන්න යෝජිත කාර්යය කරන Python වැඩසටහනක්:  ```python def get_validated_input():     '''Get validated input from the user and returns a list of integers'''     input_str = input("Enter a list of comma-s
- `9db29ab97bdf…` — ඔව්, මෙන්න වැඩසටහනේ අලුත් සංස්කරණයක්, විශේෂ දෝෂ පණිවිඩ එක්ක:  ```python def get_validated_input():     '''Get validated input from the user and returns a list of integers'''     input_str = input("Ent

**Degenerate-repetition messages** (showing 6):

- `60281c4d3184…` — අමුද්‍රව්‍ය එකට කලවම් කරන්න. තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තැටියක තෙල් තෙල් තෙල් තෙල් තෙල් තෙ
- `76c5a884ecbc…` — 1. සෞඛ්‍යය පස සංරක්ෂණය: පස කාන්දුවීම පාලනය කිරීම සහ පස සෞඛ්‍යය ප්‍රවර්ධනය කිරීම සඳහා වනාන්තරකරණය, කෘෂි වනාන්තරකරණය සහ සමෝච්ඡ කෘෂිකර්මාන්තය වැනි පස සංරක්ෂණ ක්‍රමවේදයන් ක්‍රියාත්මක කිරීම.  2. සෞඛ්‍යය වැ
- `ee8b41a64638…` — ඔව්, මම හිතන්නේ ෆ්‍රෑන්ක් නැතිවුනොත් අනිත් චරිත ගොඩක් වෙනස් වෙයි. ෆ්‍රෑන්ක් ගේ ගලාගර් පවුලේ පැවැත්ම ඔහුගේ සියලුම දරුවන්ගේ ජීවිත වලට බලපෑම් ඇති කර ඇති අතර ඔහුගේ බලපෑම ඔවුන්ගේ පෞරුෂයන්, හැසිරීම් සහ එකින
- `c3cc7890dab2…` — ඔව්, හින්දු ජ්‍යොතිෂයට අනුව, සිකුරු යාන්ත් රාව ස්ථාපිත කිරීමට හා නමස්කාර කිරීමට හොඳම දිනය හා වේලාව සිකුරාදා දිනය වන අතර එය චන්ද්‍රයාගේ වැඩිවීමේ අවධිය හෝ ශුක්ලා පාක්ෂයයි. සිකුරාදා සිකුරු දිනය ලෙස සැලකෙ
- `354517353d4c…` — ගඟේ පැත්තේ මුල් බැසගත් ගස සහ නිම්නයේ ලිලී යන දෙකම ජීවිතයේ ගමන සඳහා උපමා ලෙස භාවිතා වේ. ගඟක් අසල මුල් බැසගත් ගසක් මෙන්, ජලයෙන් පෝෂණය ලබා ගනිමින් වර්ධනය වී ශක්තිමත් වන්න, ජීවිතයේ ගමන කෙළින් හා පහසු මාවත
- `23728fae2922…` — පෙරවදන  දුප්පත්කම අවම කිරීම ලොව පුරා මිලියන ගණනක් ජනතාවට බලපාන වඩාත්ම හදිසි ගෝලීය ගැටලුවකි. දුප්පත්කම මිනිස් ජීවිතයේ සෑම අංශයකටම බලපාන අතර ආර්ථික වර්ධනයට, දේශපාලන ස්ථාවරත්වයට හා සමාජ සංවර්ධනයට බාධාවක්

**Empty turns** (showing 6):

- `38e7a263fb17…` — <empty turn>
- `7b3660a37765…` — <empty turn>
- `6a581e902db7…` — <empty turn>
- `d7d78c5b3d5c…` — <empty turn>
- `f8210c2740bf…` — <empty turn>
- `e58946390bcf…` — <empty turn>

## Limitations

- No reference translations, so this measures **dataset hygiene and fluency proxies**, not adequacy/accuracy (no BLEU/COMET vs a gold set).
- Source-aligned target/source length ratios (truncation/expansion detection) are not yet included; they require joining to the English parquet by `prompt_id`.
- The repetition and language thresholds are heuristics; the per-category example lists above are provided so the thresholds can be eyeballed and tuned.

