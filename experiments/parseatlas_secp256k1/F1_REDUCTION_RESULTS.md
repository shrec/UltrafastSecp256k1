# F1 — ერთი სეკვენციური ჯაჭვის საბოლოო პასუხის გარდაქმნა

2026-09-05. Branch `experiment/representation-search`;
HEAD `fef231d4e4173bd016fb2a3a1eff67087396a203`.
კვლევითი შედეგი მიმოხილულია; production ჩანაცვლება არ მომხდარა.

## მთავარი შედეგი

**იმავე ერთი final-only მოდულური ჯაჭვის ბლოკური მრავალბირთვიანი გარდაქმნა
დიდ ნაკრებებზე აჩქარდა**, ყველა საჭირო სამუშაოს ჩათვლით:

- 65,536 RHS / 2 MiB: **1.946×** პირველ სერიაში, **1.981×** მეორეში.
- 1,048,576 RHS / 32 MiB: **4.235×** პირველ სერიაში, **4.410×** მეორეში.
- ორივე დიდი ზომის ორივე სერიაში რვავე same-round შედარება მრავალბირთვიანის
  სასარგებლოა: ჯამში 32/32. ეს აღწერითი რაოდენობაა, არა გაერთიანებული estimate.
- 256 და 4,096 RHS-ზე cold მრავალბირთვიანი გზა მკვეთრად ნელია.
  ზუსტი crossover არ გაზომილა; მხოლოდ ამ რეგისტრირებული ზომების სურათი გვაქვს.
- ერთბირთვიანი Chunk4-ის მოგება არ გამეორდა. სრული allocation-ით Tree
  ყველა უჯრედში ნელია: paired-median კოეფიციენტის შებრუნებით დაახლოებით
  1.36–3.06-ჯერ მეტი დრო.

კოეფიციენტები არის same-round **ნორმალიზებული region-mean ხარჯების**
შეფარდებების მედიანები. CPU რესურსი არ არის ერთნაირი: Serial, Chunk4 და Tree
ერთ ბირთვზე მუშაობს; ColdMulti იყენებს რვა worker-ბირთვს და coordinator-ს.
ეს **wall-time მოგებაა, არა რვა ბირთვის უფასო ეკვივალენტი, ნაკლები ჯამური CPU
სამუშაო/ენერგიის მტკიცება ან 4.4× მთელი ენჯინის აჩქარება**.

ნაკადების შექმნა, CPU-ზე მიბმა, ყველა input-ის დამუშავება, partial state,
გაერთიანება, join, allocation/free და საბოლოო პასუხი დროში შედის.
ყველა მეთოდი იყენებს უცვლელ Original მოდულურ შეკრებას; ახალი carry ალგორითმი
ამ ცდას არ შერევია.

## რა პასუხობს მფლობელის G0 კითხვას

ობიექტია x_(i+1)=(x_i+rhs_i) mod n, canonical x0 და წინასწარ მოცემული,
ერთმანეთისგან დამოუკიდებელი RHS-ებით. საჭიროა მხოლოდ საბოლოო canonical x_N.
ეს scalar order n-ია, არა coordinate field prime p.

G0 კონტროლი ინახავს ერთ accumulator-ს და მას სეკვენციურად აახლებს.
ალტერნატიული state ინახავს **იმავე stream-ის contiguous ბლოკების ჯამებს**.
decode/merge აერთიანებს მათ და x0-ს ზუსტად ერთხელ უმატებს:
x_N = (x0 + S_0 + ... + S_(k-1)) mod n.
Canonical modular addition-ის closure და associativity ამ გარდაქმნის
ალგებრული საფუძველია. ყველა high carry და თითოეული addition-ის reduction
უცვლელ შემოწმებულ primitive-ში რჩება.

ამრიგად, წინა ექსპერიმენტის რვა დამოუკიდებელი საბოლოო accumulator-ის ნაცვლად
აქ ბლოკები **ერთი მოთხოვნილი პასუხის** წარმოდგენაა. ეს არის შეზღუდული
ობიექტის კონკრეტული executable witness იმისა, რომ საწყისი fold-ის dependency
structure არ არის ერთადერთი კანონიერი გამოთვლის სტრუქტურა.

მაგრამ intermediate prefixes, შუალედური side effects ან წინა პასუხზე
დამოკიდებული მომდევნო input ამ კონტრაქტში არ შედის. მათთვის ეს F1 შედეგი
პირდაპირ არ გადადის. არც ცნობილი associativity/parallel reduction გამოცხადებულა
ახალ მათემატიკურ აღმოჩენად.

## სრული შედარების მატრიცა

`Serial / candidate` > 1 კანდიდატის სასარგებლოა. თითო უჯრედში 8 region mean;
სერიები არ გაერთიანებულა. ბოლო სვეტი ოთხივე მეთოდის 32 measured ჩანაწერში
200 ms-ზე ნაკლები region-ების რაოდენობაა.

| სერია | RHS რაოდენობა | RHS ზომა | Serial / Chunk4 | Serial / Tree | Serial / ColdMulti | <200 ms |
|---|---:|---|---:|---:|---:|---:|
| 1 | 256 | 8 KiB | 1.0117 | 0.7342 | 0.0019 | 24/32 |
| 1 | 4,096 | 128 KiB | 0.9940 | 0.6792 | 0.0853 | 16/32 |
| 1 | 65,536 | 2 MiB | 0.9505 | 0.6458 | 1.9461 | 0/32 |
| 1 | 1,048,576 | 32 MiB | 0.9799 | 0.3272 | 4.2351 | 0/32 |
| 2 | 1,048,576 | 32 MiB | 0.9561 | 0.3326 | 4.4100 | 0/32 |
| 2 | 65,536 | 2 MiB | 0.9475 | 0.6431 | 1.9809 | 0/32 |
| 2 | 4,096 | 128 KiB | 0.9848 | 0.6703 | 0.0772 | 0/32 |
| 2 | 256 | 8 KiB | 0.9528 | 0.7166 | 0.0051 | 0/32 |

ქვემოთ დრო არის **µs / სრული job**, რვა region mean-ის მედიანა.
ეს individual-job latency-ის განაწილების მედიანა არ არის.

| სერია | RHS | Serial | Chunk4 | Tree + allocation | ColdMulti |
|---|---:|---:|---:|---:|---:|
| 1 | 256 | 1.499 | 1.503 | 2.064 | 838.551 |
| 1 | 4,096 | 17.403 | 17.521 | 25.975 | 209.194 |
| 1 | 65,536 | 581.511 | 608.994 | 900.227 | 297.413 |
| 1 | 1,048,576 | 10047.341 | 10422.996 | 30971.047 | 2378.040 |
| 2 | 1,048,576 | 10925.165 | 10603.144 | 30419.931 | 2415.339 |
| 2 | 65,536 | 571.928 | 604.834 | 892.768 | 291.805 |
| 2 | 4,096 | 17.204 | 17.474 | 25.665 | 223.038 |
| 2 | 256 | 1.079 | 1.131 | 1.504 | 212.935 |

მნიშვნელოვანი მაგალითი: მეორე სერიის 1,048,576 RHS-ზე Chunk4-ის marginal
მედიანა Serial-ზე ნაკლებია (10.603 ms vs 10.925 ms), მაგრამ წინასწარ არჩეული
same-round ratio-ების მედიანა **0.9561**-ია და მხოლოდ 3/8 წყვილი იგებს.
მოგების მისაღებად სტატისტიკური საზომი არ შეგვიცვლია.

Chunk4-ს სულ 13/64 ხელსაყრელი წყვილი აქვს, Tree-ს 0/64, ColdMulti-ს 32/64 —
ყველა დიდი-N უჯრედებიდან. ეს pair-count-ებია, არა pooled performance estimate.
ყველა outlier, min/max/MAD/inclusive-IQR და თითო round-ის ratio raw JSON-შია.

## დამატებითი სამუშაო და locality-ის ფასი

| წარმოდგენა, N>0 | source modular additions / job | სამუშაო მეხსიერება / scheduling |
|---|---:|---|
| Serial | N | ერთი სეკვენციური accumulator, allocation არა |
| Chunk4 | N+4 | ოთხი contiguous ბლოკი, ერთ ბირთვზე მონაცვლეობით, stack partials და merge |
| Tree | N | ყოველ job-ზე N×32 ბაიტის vector allocation, zero-init, სრული copy, in-place pair merge და free |
| ColdMulti | N+k; აქ k=8 | k zero-origin ბლოკი, partial/error/thread vectors, create/pin/join და ordered merge |

ColdMulti-ს მეტი source addition აქვს და მაინც ნაკლები wall time დიდ N-ზე.
ეს აჩვენებს, რომ ოპერაციების რაოდენობა მარტო საკმარისი ranking არ არის.
მაგრამ ამ მოგების მიზეზად DRAM bandwidth-ის შემცირება არ დამტკიცებულა:
მრავალბირთვიანი რესურსიც შეიცვალა და PMU traffic counters არ შეგვიკრიბავს.

Tree-ს უფრო დაბალანსებული **მათემატიკური** reduction structure
არ ნიშნავს ამ binary-ში logarithmic parallel span-ს. In-place compaction
storage anti-dependencies-ს ქმნის; ეს tree implementation ერთ ბირთვზეა.
მისი memory/allocator ხარჯი რეალურად ჩართულია და შემდგომ reusable-workspace
ან სხვა layout-ის ცდა ცალკე რეგისტრირებულ ვარიანტს მოითხოვს.

Source zero/copy/add/thread counts არ არის dynamic instruction, physical byte,
allocator syscall ან ენერგიის მრიცხველები. Compiler packing/stores და
allocation-ის განმეორებითი reuse შედეგის ნაწილია.

## სისწორის gates და audit

- Root-მა გაუშვა 5 C++ correctness კონფიგურაცია: GCC 14.2.0 native,
  GCC portable carry, Clang 18.1.3, GCC UBSan და GCC ASan+UBSan.
  ყველა pass; იგივე კორპუსი ხუთ დამოუკიდებელ კორპუსად არ ითვლება.
- თითო კონფიგურაცია: **4,971 fixture** (360 boundary, 515 targeted,
  4,096 random), 697,896 RHS / რეალური Scalar replay addition.
  Fixture ოჯახებს გადაფარვა შეიძლება ჰქონდეს.
- თითო fixture-ზე Serial, Chunk4, Tree და random contiguous partition fold
  შედარდა დამოუკიდებელ Boost cpp_int-სა და უცვლელი public Scalar +=-ის
  საბოლოო პასუხს: სრული 32 LE ბაიტი, canonicality და public BE encoding.
- Multicore გამიჯნული subset-ია: **53 fixture / 159 call**.
  Correctness CPU სიებია 1/6/12 logical CPU აქ; performance-ის
  8 physical-core სიასთან ეს სხვა test matrix-ია. JSON ამას ცალ-ცალკე ასახავს.
- შემოწმდა N=0/1, odd tail, არათანაბარი/ცარიელი ბლოკები, N<worker list,
  n-ისა და 2^256-ის საზღვრები, input preservation, scratch canaries,
  19 checked error, 6 დამატებითი empty identity და 256 bit-corruption control.
- CPU 1023-ის disposable affinity probe დაბრუნდა errno 22-ით.
  valid-first/failing-second worker სცენარმა შესაბამისი exception გაავრცელა;
  caller affinity და input-ები უცვლელია, live threads 1→1.
  Thread creation failure ხელოვნურად არ გამოწვეულა; მისი cleanup code/assembly-ით
  მიმოხილულია. ASan/UBSan არ არის TSan ან race-freedom proof.
- რვა benchmark smoke: native/UBSan × ოთხი ზომა, pass.
  ათი უარყოფითი driver CLI და correctness-ის ზედმეტი არგუმენტი exit 2-ით უარყოფილია.
- Root mechanical gates-ის შემდეგ სრულად წაიკითხა header, tests, driver და
  protocol. დამატებით დამოუკიდებელმა reviewer-მა source და binary შეამოწმა:
  ოთხი timed outer loop, ოთხი full-job routine, chunk/multicore/worker paths.
- Binary ინარჩუნებს ყოველ job-ს, input-ებს, reduction-ს და 32-byte writeback-ს.
  Tree allocation/init/copy/free და multicore startup/affinity/join/error cleanup
  clocks-ის შიგნითაა. 17 ზუსტი disassembly range შენახულია.
- **256 measured + 64 warmup + 32 preflight + 86 calibration** ჩანაწერი.
  Root და დამოუკიდებელი off-clock JS audit: 0 შეუსაბამობა source work,
  job caps, chronology, seeded order, checksums, topology და statistics-ში.
  Unix ns raw ტექსტად შენარჩუნებულია, audit-ში BigInt გამოიყენება.

ყოველი job სრულ 32-ბაიტიან პასუხს materialize-ებს. თითო recorded region-ის
**ბოლო** იდენტური job-ის შედეგი სრულად მოწმდება; ყველა შიდა job ცალ-ცალკე
არ შემოწმებულა და ყველა intermediate prefix-ის შემოწმებაც არ გვიცხადებია.
Raw checksum-ის თანხვედრა ცალკე byte-equivalence proof არ არის;
სრული ბაიტური შედარება native C++ gate-ში ხდება.

## კალიბრაციის სუსტი ადგილი და შენარჩუნებული უარყოფითი მონაცემები

Cold thread startup პატარა N-ზე ძვირია, ამიტომ თითო მეთოდს თავისი
job-count კალიბრაცია აქვს. Same-round ratio ადარებს elapsed/jobs მნიშვნელობებს
**განსხვავებული სამუშაო მოცულობებით**, არა matched-work latency-ს.
ეს normalization cache/allocator/scheduler/thermal განსხვავებას არ აუქმებს.

ყველა საბოლოო calibration trial მიაღწია ≥200 ms-ს, მაგრამ მოგვიანებით
**40/256 measured region ამ ზღვარს ჩამოსცდა**:

- სერია 1, N=256: Serial 8, Tree 8, ColdMulti 8 — სულ 24.
- სერია 1, N=4096: Serial 5, Chunk4 8, ColdMulti 3 — სულ 16.
- დანარჩენ ექვს invocation-ში 0; ორივე დიდი ზომის ორივე სერიაში 0.

საერთო measured დიაპაზონია **9.768472–357.034367 ms**.
მაგალითად პირველი N=256 ColdMulti calibration აირჩია 38 job;
მისი ბოლო calibration 323.709394 ms იყო, მაგრამ ყველაზე მოკლე measured
region 9.768472 ms გამოვიდა. ეს კალიბრაციის/შემდგომი გარემოს სტაბილურობის
რეალური შეზღუდვაა. მიზეზი არ არის იზოლირებული. არც ერთი ჩანაწერი არ ამოგვიღია,
არ შეგვიცვლია source/plan და ხელსაყრელი rerun არ აგვირჩევია.

ამიტომ მცირე-N უჯრედებში პატარა პროცენტული სხვაობა გამარჯვების მტკიცებად
არ გამოიყენება; შემდგომი precision study წინასწარ ახალ stabilization/adaptive-duration
წესს და ყველა მცდელობის შენახვას მოითხოვს. დიდი-N მოგების ოთხივე უჯრედში
200 ms-ის მიზანი სრულად შესრულდა, თუმცა დანარჩენი გარემოს caveat-ები მაინც რჩება.

## გარემო და გადატანის საზღვარი

Intel i5-14400F: 16 logical CPU, 10 physical core.
Coordinator CPU 4-ზეა; მისი sibling CPU 5-ია.
Worker CPU-ებია **0,2,6,8,10,12,13,14**: თითო allowed physical core-ის
ერთი წარმომადგენელი. Coordinator-ის მთელი core და დამატებითი core
(წარმომადგენელი CPU 15) headroom-ად გამოირიცხა.
ეს ჩვენი workload-ის არჩევანია; OS-სა და გარე პროცესებს core არ დაჯავშნიათ.
Core-ები ჰეტეროგენულია; worker count დაკვირვებული topology-დან გამოვიდა.

ყველა endpoint-ზე governor `performance`, no_turbo `1`; frequency snapshot-ები
განსხვავდება, ზოგჯერ 800,000 kHz-მდეც ჩამოდის. **უწყვეტი ან ფიქსირებული სიხშირე
არ გაზომილა**. Root-ს sudo ან გლობალური CPU setting არ შეუცვლია.
ჩვენივე builds/tests/benchmarks გაზომვის დროს შეჩერებული იყო;
გარე დატვირთვა, sibling activity და thermals იზოლირებული არ ყოფილა.

ყოველ invocation-ში იგივე runtime-generated canonical x0/RHS ყველა job-ში
ხელახლა იკითხება. ეს განმეორებითი immutable-input რეჟიმია; cache,
branch predictor, allocator და OS thread resources შეიძლება გათბეს.
„Cold“ მხოლოდ ყოველ job-ზე thread lifecycle-ს ნიშნავს.
Input generation/reference conversion ყველა მეთოდისთვის clocks-ის გარეთაა.
წინასწარი algebraic summary არც ერთ კანდიდატს არ მიუღია.

სამი live Source Graph caller დაკვირვება (წყაროს hash manifest-შია):

- [MuSig2 aggregation](../../src/cpu/src/musig2.cpp), `musig2_partial_sig_agg`,
  lines 614–645: საბოლოო scalar sum, მაგრამ **ct::scalar_add** და tweak/degeneracy
  checks აქვს; კომენტარი პირდაპირ მიუთითებს secret contributions-ის CT-002 ფიქსზე.
- [FROST aggregation](../../src/cpu/src/frost.cpp), `frost_aggregate`,
  lines 649–692: scalar sum-თან ერთად group commitment და validation;
  აქაც **ct::scalar_add** გამოიყენება.
- [Pedersen blind sum](../../src/cpu/src/pedersen.cpp), `pedersen_blind_sum`,
  lines 131–143: შემავალი blinds-ის plus და გამომავალი blinds-ის minus ეტაპები.
  ეს ჯერ ჩვენი F1 array contract-ის სრული production adaptation არ არის.

ამ caller-ების რეალური ზომების განაწილება და end-to-end contribution არ გაგვიზომავს.
**Variable-time F1 არ ჩაანაცვლებს CT caller-ს** ბაიტური თანხვედრისა და benchmark-ის
საფუძველზე. არც multiplication/inversion/group operation/signature acceleration
და არც მთელი Secp256k1 ენჯინის მოგება ამ ცდით არ იზომება.

## 25 ლინზის delta და შემდეგი ნაბიჯი

ეს დამატება მიჰყვება [არსებულ 25-ლინზიან რუკას](L0_25_LENSES.md);
ძველი hash-bound ledger არ გადაგვიწერია. ეს არ არის დასრულებული 25×25 ძიება.

| ლინზა | F1-ში დანახული / დარჩენილი |
|---|---|
| V01 reachability | იგივე final-state fold → contiguous summaries/tree კანონიერი კონკრეტული witness; თავისუფალი გენერატორი არა |
| V02 corpus | 4,971 fixture და multicore subset, დიდი-N native API preflight; კორპუსის საზღვარი მკაფიოა |
| V03 equivalence | closure/associativity არგუმენტი + სრული ბაიტური finite checks; compiled full-domain proof არა |
| V04 definedness | canonical/extent/disjointness preconditions და checked error gates |
| V05 range | ყოველ primitive-ზე canonical mod-n state; wide/lazy accumulator ჯერ არ დამატებულა |
| V06 alias | const inputs, exact preservation, scratch canaries; სხვა overlap contract დაუდგენელია |
| V07 observable | ერთი საბოლოო 32-byte შედეგი; ყველა prefix და online side effects არა |
| V08 invariants | x0 ერთხელ, თითო operand ერთხელ თითო job-ში, ბლოკები სრულ input-ს ფარავს |
| V09 CT | fast variable-time; CT caller adaptation და secret scratch handling ცალკე gate-ია |
| V10 resources | jobs/inputs/thread-launch caps, cleanup და observed-core headroom |
| V11 property visibility | ასოციაციური block summary გამოავლენს კანონიერ decomposition-ს; ახალი extractor არა |
| V12 depth | abstract fold/tree განსხვავება; in-place storage-ით compiled logarithmic span არ მტკიცდება |
| V13 live state | ერთი accumulator / ოთხი partial / N scratch / k worker partials; რეალური allocator/codegen ჩართულია |
| V14 operations | ColdMulti N+8-ითაც იგებს დიდ N-ზე; static count მარტო ranking ვერ არის |
| V15 conversion | Tree copy/init/free და block merge შედის; ახალი რადიქსი/CRT/Montgomery არა |
| V16 latency | სრული job-ის region mean, არა isolated scalar-add ან individual-job distribution |
| V17 throughput | იგივე ერთი საბოლოო პასუხის internal multicore execution; რესურსის ფასი განსხვავდება |
| V18 instructions | 17 disassembly range gate; dynamic PMU counters არა |
| V19 cache | 8 KiB..32 MiB, explicit scratch, განმეორებითი input; physical bandwidth მიზეზი დაუდგენელია |
| V20 code size | hashed wrappers და reachable bodies; shared code ორჯერ არ ითვლება |
| V21 portability | GCC/Clang/portable/sanitizers ერთ Linux x86 host-ზე; სხვა არქიტექტურა არა |
| V22 compiler | performance GCC native-ზე; მეორე compiler მხოლოდ correctness |
| V23 policy/ties | ორი წინასწარ არჩეული სერია და balanced order; ყველა representation policy-ის sensitivity არა |
| V24 transfer | ერთი ჯაჭვის final-only საზღვარზე დიდი-N მოგება; CT/რეალური caller/other primitives ღიაა |
| V25 prior art | ცნობილი modular reduction/associativity კონტროლები; novelty არ ცხადდება |

შემდეგი ძირითადი კვლევა **F2 — bounded delayed normalization**:
იმავე final-only პასუხისთვის wide/column state-ის ზუსტი ზრდის ზღვარი,
summary/merge და საბოლოო mod-n reduction უნდა შევადაროთ eager normalization-ს.
მიზანია გავიგოთ, შეიძლება თუ არა ერთი ბირთვის სრული სამუშაოც გაიაფდეს —
არა მხოლოდ მეტი ბირთვის გამოყენებით wall time-ის შემცირება.

მის შემდეგ **P1 — ყველა canonical prefix-ის scan**, მთელი output traffic-ით.
CT caller-ის adaptation, reusable tree workspace, worker-count/crossover და
კალიბრაციის stabilization ცალკე წინასწარ რეგისტრირებული ჭრილებია.
არც F2, არც P1, არც production replacement ამ snapshot-ში განხორციელებული არ არის.

## არტეფაქტები

- [კონტრაქტი და პროტოკოლი](F1_REDUCTION_PROTOCOL.md),
  [kernel](probes/f1_reduce.hpp), [C++ tests](tests/test_f1_reduce.cpp),
  [driver](probes/f1_reduce_compare.cpp).
- [წინასწარი გეგმა](data/f1_run_plan_20260905.json):
  2026-09-05 16:13:47 UTC; SHA256
  `eb4a8042016973e175e3a83361018197099fa6864486dca1a30cd6401dcf1f19`.
  `measurements_started:false` რეგისტრაციის უცვლელი snapshot-ია.
- [Validation record](data/f1_validation_20260905.json):
  build commands/results, ხუთი correctness output, რვა raw smoke,
  უარყოფითი CLI, გარემო და whitespace receipts.
- [Numeric audit](data/f1_numeric_audit_20260905.json):
  ყველა უჯრედის შედეგი და replayable off-clock JS audit function.
- [Disassembly](data/f1_disassembly_20260905.txt), 17 exact range.
- [Manifest](data/f1_manifest_20260905.json): source/binary/raw/report hashes,
  command provenance, დამოუკიდებელი review და ყველა რვა raw JSON-ის path.
- Measured binary `/tmp/parseatlas-f1-review.pkv6Nn/f1_compare`,
  SHA256 `6db8685925eceb4d10be15458bf08c34c3efc700d2656e79e5c9de0a345e8ed0`.
  Temporary binary მუდმივი archive არაა; code/commands/hash/assembly შენახულია.

Task MCP მფლობელის მითითებით კვლავ დროებით გამოტოვებულია;
`PA_SECP_M1_MODN_NATIVE_015` არ გაშვებულა/მიღებულა/დახურულა.
Session `01a06be6-2904-7c62-9e7d-1245c34a5312` შენარჩუნებულია.
Production/reference/default build/CI და ძველი evidence ledger უცვლელია;
commit/push არ გაკეთებულა. Frozen protocol-ის line 227-ზე მხოლოდ cosmetic
extra EOF blank-line warning ინახება; მისი preregistered hash არ შეცვლილა.
