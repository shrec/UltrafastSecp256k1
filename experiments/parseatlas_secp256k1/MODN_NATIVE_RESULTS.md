# Scalar-order n: სრული მოდულური შეკრების native შედეგები

2026-09-05. კვლევითი snapshot; production acceptance **არა**.
Branch: `experiment/representation-search`; HEAD: `fef231d4e4173bd016fb2a3a1eff67087396a203`.

## შედეგი და გადაწყვეტილება

**ამ კონკრეტულ სრულ mod-n workload-ში Blocked2 და GP-recomputed ვერ აჯობებს
იგივე inline scaffold-ში Original-ს:** ორივე სერიის ოთხივე უჯრედში მათი
paired-median Original/candidate კოეფიციენტი 1-ზე ნაკლებია.
ეს არის გაზომილი უარყოფითი შედეგი ორი კანდიდატისთვის, არა წარმოდგენების
მეთოდოლოგიის ან ყველა შესაძლო პარალელური გარდაქმნის უარყოფა.

Original/candidate paired median-ის შებრუნებით დათვლილი დამატებითი დრო:
Blocked2 დაახლოებით **1.7–8.1%**, GP-recomputed **11.5–16.9%**.
ეს უჯრედების დიაპაზონია, არა confidence interval ან მთელი ენჯინის შეფასება.
წინა carry-მიკროტესტის უპირატესობა ამ სრულ საზღვარზე მოგებად არ გადმოვიდა;
მიზეზის იზოლირება ამ ცდით დასრულებული არ არის.

მიმდინარე production კოდი, default build და CI უცვლელია. ჩანაცვლების
უფლებამოსილება გვაქვს; ამ კანდიდატების ჩანაცვლებისთვის დადებითი მონაცემი არ გვაქვს.
კვლევის შემდეგი საზღვარია **იმავე დამოკიდებული ამოცანის** წარმოდგენით
გარდაქმნა, სრული encode/merge/normalize/decode ხარჯით.

## რა გაიზომა

სამი matched inline გზა: Original, Blocked2, GP-recomputed.
ყველას აქვს ერთი და იგივე canonical [0,n) შესვლა, სრული 256-ბიტიანი raw sum
და high carry, რეალური fast Scalar-ის early-return/ORDER-subtraction სქემა
და canonical (a+b) mod n გამოსვლა. ეს scalar order n-ია, არა field modulus p.
მეოთხე გზა უცვლელ public Scalar::operator+=-ს ცალკე scalar.cpp TU-დან იძახებს
(no LTO). იგი მხოლოდ API დიაგნოსტიკაა: დამატებითი ფუნქციური საზღვრების გამო
მისი inline კანდიდატთან შეფარდება carry-ალგებრის აჩქარებად არ ითვლება.

ერთი ჯაჭვი დამოკიდებული recurrence-ია. Throughput ფორმა იმავე RHS ნაკადს
რვა დამოუკიდებელ accumulator-ზე ანაწილებს; ეს **ერთი ჯაჭვის პარალელური
გარდაქმნა არ არის** და განსხვავებული სრული state აქვს.
Count არის თითო pass-ის ჯამური შეკრებების რაოდენობა, არა თითო lane-ის.
256 და 4096 canonical RHS ელემენტი შესაბამისად 8 და 128 KiB-ია თითო
წარმოდგენაში. Limbs და Scalar წარმოდგენები ერთად არსებობს; timed region-ში
მხოლოდ არჩეული გამოიყენება. არც ერთი ზომა არ ნიშნავს გაზომილ DRAM traffic-ს.

## დროები

ყველა დრო ns / სრული მოდულური შეკრებაა, თითო უჯრედში 8 ნიმუშის მედიანა.
ბოლო ორი სვეტი same-round Original/candidate შეფარდებების მედიანებია;
**1-ზე მეტი იქნებოდა კანდიდატის უპირატესობა**. სერიები არ გაერთიანებულა.
API სვეტი მხოლოდ დიაგნოსტიკურია.

| სერია | ფორმა | RHS | Original | Blocked2 | GP recomputed | actual API (დიაგნ.) | Original / Blocked2 | Original / GP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ერთი ჯაჭვი | 8 KiB | 13.718 | 14.839 | 15.676 | 18.110 | 0.9248 | 0.8749 |
| 1 | 8 დამოუკიდებელი ჯაჭვი | 128 KiB | 10.914 | 11.140 | 12.570 | 15.532 | 0.9800 | 0.8685 |
| 1 | ერთი ჯაჭვი | 128 KiB | 16.578 | 17.252 | 18.477 | 20.997 | 0.9620 | 0.8965 |
| 1 | 8 დამოუკიდებელი ჯაჭვი | 8 KiB | 9.971 | 10.360 | 11.648 | 14.874 | 0.9617 | 0.8554 |
| 2 | 8 დამოუკიდებელი ჯაჭვი | 8 KiB | 9.966 | 10.688 | 11.637 | 15.059 | 0.9612 | 0.8573 |
| 2 | ერთი ჯაჭვი | 128 KiB | 16.588 | 17.283 | 18.507 | 21.025 | 0.9619 | 0.8970 |
| 2 | 8 დამოუკიდებელი ჯაჭვი | 128 KiB | 10.934 | 11.138 | 12.552 | 15.538 | 0.9831 | 0.8708 |
| 2 | ერთი ჯაჭვი | 8 KiB | 14.095 | 14.874 | 16.043 | 18.434 | 0.9475 | 0.8783 |

ყველა calibration, warmup და measured ნიმუში შენახულია, გამონაკლისების ამოღების
გარეშე. განსაკუთრებით ხმაურიანია throughput/8 KiB: პირველი სერიის GP
მაქსიმუმი 15.717 ns-ია, API-ის 21.198 ns; მეორე სერიის Original მაქსიმუმი
14.668 ns-ია, Blocked2-ის 15.070 ns. ცალკეული paired ratio ზოგჯერ 1-ს აჭარბებს,
მაგრამ ეს მედიანურ მოგებად არ გადაგვიქცევია. სრული min/max/MAD/inclusive-IQR
და რვავე round-ის ratio თითო raw JSON-შია.

## სისწორისა და გაზომვის gates

- ოთხი native C++ correctness კონფიგურაცია: GCC 14.2.0 native,
  GCC portable carry (`SECP256K1_NO_INT128`), Clang 18.1.3, GCC UBSan.
  ოთხივეში pass; იგივე კორპუსი ოთხ დამოუკიდებელ კორპუსად არ ითვლება.
- თითო კონფიგურაციაში 126,593 fixture შემთხვევა, მათ შორის 100,000 random pair;
  fixture ოჯახებს გადაფარვა აქვთ. 85 deduplicated boundary value და 7,225
  boundary pair; 17,280 carry-pattern და 2,088 targeted შემთხვევა.
- დამოუკიდებელი Boost cpp_int მოდული, უცვლელი public Scalar + და +=,
  კანდიდატების 32 LE ბაიტი და public BE encoding, alias x+=x, canonical
  შედეგი და input preservation შემოწმებულია. 128 recurrence chain / 32,896
  ნაბიჯი; 256 comparator bit-corruption negative control. უთანხმოება: 0.
- Correctness კორპუსში: 62,751 no-reduction, 3,761 reduction-without-wrap,
  60,081 wrap-reduction შემთხვევა. ეს ტესტების დაფარვაა, არა timed path პროფილი.
- ორივე benchmark binary-ზე 2 ფორმა × 2 ზომა smoke გაიარა; 9 ცუდი driver CLI
  და correctness binary-ის ზედმეტი არგუმენტი უარყოფილია exit 2-ით.
- Mechanical gates-ის შემდეგ სრულად მიმოხილულია header, tests, driver,
  protocol და რვავე timed wrapper-ის მიღწევადი assembly. შენარჩუნებულია
  ყველა საჭირო loop, limb/lane და 256-ბაიტიანი state writeback.
- სამ matched throughput wrapper-ს თითო region-ზე მხოლოდ ერთი outlined
  advance call აქვს; თითო add-ზე dispatch/call არ არის. რეალური API გზა
  ინარჩუნებს Scalar+= → add_impl გამოძახებებს. Packing/stack state რჩება:
  ეს არ არის spill-free ან იზოლირებული ALU benchmark.
- წინასწარ დაფიქსირდა 8 serial invocation: 2 სერია × 2 ფორმა × 2 ზომა.
  თითო invocation-ს აქვს 2 warmup და 8 measured round × 4 მეთოდი;
  seeded rotation ყველა მეთოდს ყველა პოზიციაზე ორჯერ ათავსებს.
- დასრულდა 256 measured sample, 64 warmup და 21 calibration trial.
  measured ხანგრძლივობა 222.248–509.479 ms; 200 ms-ზე ნაკლები measured sample: 0.
- ყველა warmup/measurement-ის სრული 256 LE ბაიტი შედარდა უცვლელი public API-ის
  untimed replay-ს, იგივე reset, warmup და სრული work length-ით.
  საბოლოო calibration replay-checked-ია; შუალედური calibration trials არა.
- Off-clock JavaScript audit-მა თავიდან დათვალა work, timestamps, order,
  CPU, checksum-ის თანხვედრა, statistics და paired ratios. ეს native arithmetic
  timing-ს არ ცვლის; raw Unix ns ტექსტი უცვლელია და audit-ში BigInt გამოიყენება.

## გარემო და შეზღუდვები

Intel Core i5-14400F, 16 logical CPU, 10 physical core. ერთ serial benchmark
worker-ს CPU 4-ზე affinity აქვს; CPU 5 მისი sibling-ია. ჩვენივე builds/tests
timing-მდე დასრულდა. გარე პროცესები, sibling-ის დატვირთვა და თერმული მდგომარეობა
იზოლირებული არ ყოფილა.

Governor endpoint-ებზე `performance`, `intel_pstate/no_turbo=1`.
CPU 4-ის სიხშირის snapshot გაზომვებამდე **2,500,049 kHz**, დასრულების შემდეგ
**800,000 kHz**. მეორე snapshot უკვე inactive მომენტშია; ამ ორი წაკითხვით
timed interval-ის სიხშირეს ვერ ვადგენთ. **ფიქსირებული 2.5 GHz-ის მტკიცება არ გვაქვს**.
არც sudo, არც გლობალური პარამეტრის ცვლილება არ გამოყენებულა ამ ეტაპზე.

დროები მხოლოდ GCC native binary-ს ეხება. Clang/portable/UBSan შედეგები მხოლოდ
correctness gates-ია. PMU/cycles/cache-miss/DRAM-byte counters არ შეგვიკრიბავს.
8-lane state-ის 32 scalar სიტყვა GPR ბიუჯეტს აღემატება; layout/packing/spills
შედეგის ნაწილია. უფრო დიდი RHS-ის შენელების მიზეზს მარტო bandwidth-ს ვერ მივაწერთ.

Timed input არის fixed-seed canonical stream, არა რეალური caller-ების პროფილი.
Untimed ერთი pass-ის no-carry threshold-reduction count ყველა უჯრედში 0-ია;
ეს rare path correctness-ში შემოწმებულია, მაგრამ timed coverage ამით არ მტკიცდება.
Prefix path counts არ არის გაზომვის მთელი trajectory-ის პროფილი.

Variable-time fast Scalar საზღვარი: **constant-time უსაფრთხოების მტკიცება არა**.
არც ფორმალური compiled-equivalence proof, არც novelty/prior-art შეფასება,
არც whole-engine/signature speedup და არც 25×25 სივრცის დასრულება არ ცხადდება.

## 25 ლინზის delta

იდენტობები მიჰყვება [არსებულ რუკას](L0_25_LENSES.md). ძველი hash-bound ledger არ
შეცვლილა. ქვემოთ მიმდინარე mod-n ცდის დამატებაა; სიტყვა „ნაწილობრივი“ არ არის
დასრულებული proof ან gate-ის გაუმართლებელი მიღება.

| ლინზა | ამ ცდის მტკიცებულება / დარჩენილი კითხვა |
|---|---|
| V01 მიღწევადობა | სამი ხელით მიწოდებული lawful carry template mod-n scaffold-ში; თავისუფალი ძიება არა |
| V02 კორპუსი | 126,593 fixture, recurrence და რეალური API replay; input ოჯახები შეიძლება მეორდებოდეს |
| V03 ეკვივალენტობა | a+b<2n და ერთ subtraction-ზე ალგებრული არგუმენტი + native checks; compiled proof არა |
| V04 definedness | canonical domain, unsigned arithmetic, strict CLI და UBSan; unchecked hotkernel precondition რჩება |
| V05 bounds | sum/high-carry და ერთჯერადი reduction შემოწმებულია; მრავალოპერაციული lazy state ჯერ ღიაა |
| V06 alias | x+=x და input preservation; ახალი overlapping batch contract ჯერ არა |
| V07 observable | 32-ბაიტიანი arithmetic პასუხები; harness-ში სრული 8-lane/256-byte state replay |
| V08 invariants | canonical input/output და predicate controls; სხვა representation-ის decode ჯერ დასადგენია |
| V09 constant time | არსებული variable-time გზა; secret-safe production accept დაუდასტურებელია |
| V10 resources | pass/operation/CLI bounds და serial timing; ახალი scratch/worker budget ჯერ არა |
| V11 property visibility | GP template ხელმისაწვდომია; generator/extractor-ს ახალი შედეგი არ აქვს |
| V12 dependency depth | reachable assembly loops დადასტურებულია; სრული DAG-depth/cycle profile არ გაზომილა |
| V13 live state | packing/stack materialization ჩანს; 8 lanes არაა spill-free |
| V14 operations | raw carry-ს ერთნაირი reduction დაემატა; dynamic instruction count არა |
| V15 conversion | ერთნაირი region state copy/writeback; სხვა სამყაროს encode/decode ჯერ არ იზომება |
| V16 latency | ერთი დამოკიდებული ჯაჭვის ორი RHS ზომა, ორი სერია; იზოლირებული ADC latency არა |
| V17 throughput | რვა დამოუკიდებელი ჯაჭვი; იგივე ერთი recurrence-ის parallelization არა |
| V18 instructions | 13 exact disassembly range მიმოხილულია; შიდა PMU counters არა |
| V19 cache | 8/128 KiB RHS რეჟიმები; residency/traffic/DRAM მიზეზი დაუდგენელია |
| V20 code size | ზუსტი wrapper/outlined-body ზომები და hash; shared code-ის ორმაგი დათვლა არ უნდა მოხდეს |
| V21 portability | GCC native/portable და Clang correctness ერთ x86 host-ზე; სხვა არქიტექტურა არა |
| V22 compiler | 2 compiler correctness; performance მხოლოდ GCC 14.2.0-ზე |
| V23 policy/ties | deterministic balanced order; representation-policy sensitivity კვლევა არ ჩატარებულა |
| V24 transfer | raw-carry კანდიდატების მოგება full mod-n-ში არ დადასტურდა; კონკრეტული უარყოფითი transfer შედეგი |
| V25 prior art | novelty არ ფასდება; წარმოდგენით პარალელიზმის შემდგომი ცდა არ უნდა მოინათლოს უცნობ აღმოჩენად |

## შემდეგი კონტრაქტი: G0-ში სეკვენციური, სხვა წარმოდგენაში პარალელიზებადი

მფლობელის საკვანძო კითხვა ასე ფიქსირდება: დამოკიდებულება თვითონ ამოცანის
სავალდებულო დაკვირვებად ქცევას ეკუთვნის, თუ მხოლოდ საწყის წარმოდგენას G0-ში?
G0 აქ საწყისი წარმოდგენის სახელია, არა ახალი ფორმალური კლასიფიკაციის მტკიცება.

მიმდინარე 8-chain workload უკვე დამოუკიდებელ ამოცანებს აერთიანებს.
შემდეგი ცდა კი **იგივე ერთი recurrence-ის** x_i=(x_(i-1)+a_i) mod n კონტრაქტს
ინარჩუნებს და ორ ერთმანეთისგან მკაფიოდ გამოყოფილ observable-ს განიხილავს:

1. **Final-only:** საჭიროა მხოლოდ x_N. იგივე stream შეიძლება ბლოკურ modular
   summaries-ად დაიშალოს და გაერთიანდეს. ყველა input ერთხელ runtime-ში უნდა
   დამუშავდეს; baseline-საც იგივე precomputation უფლება აქვს. მიმდინარე
   benchmark-ის განმეორებადი stream-ის წინასწარ შეჯამებას ზოგად online
   recurrence-ზე მოგებად ვერ გავასაღებთ.
2. **All-prefix:** საჭიროა ყველა canonical x_i. ბლოკის local prefixes,
   block-total scan და offset propagation output-ის იმავე რიგს აღადგენს.
   სრული output materialization და scratch traffic ორივე მხარეს დროის ნაწილია.
   თუ შემდეგი input წინა output-ზეა დამოკიდებული ან გვერდითი ეფექტი არსებობს,
   ეს დამოუკიდებელი-input scan contract აღარ არის.
3. **Inner-limb carry:** იგივე ერთი სრული addition-ის carry-transfer summaries
   კომპოზიცია შეიძლება tree/blocked schedule-ად წარმოვადგინოთ. carry summary
   მარტო low result-ს არ ინახავს; sum state და decode აუცილებელია.
   მცირე ოთხ-limb შემთხვევაში შემცირებული თეორიული depth თავისთავად სისწრაფე არაა.

H03-ის გაფართოებული/lazy representation მხოლოდ სრული range witness-ით დაიშვება:
high carry-ის დაკარგვა დაუშვებელია; reduction-ის გადადებისას რამდენიმე operand-ის
ჯამს ერთი high bit ყოველთვის ვერ იტევს. H04-ის layout ცდა ცალკე ითვლის
AoS/SoA/packing/unpacking-ის ფასს და უცვლელ workload-ს ინარჩუნებს.

მომდევნო რიგია **F1 modular block reduction → F2 bounded delayed normalization
→ P1 all-prefix scan**; C1 inner-limb carry study ცალკე საზღვარია.
F2-ში N RHS-სა და x_0-ის ჯამი (N+1)·2^256-ზე ნაკლებია: ზოგადი safe storage
bound არის 256+ceil(log2(N+1)) ბიტი. ფართო ჯამზე მიმდინარე ერთჯერადი
ORDER-subtraction reducer-ის გამოყენება დაუშვებელია ახალი reduction proof-ის გარეშე.
განმეორებადი stream-ის შემთხვევაში final-only ალტერნატივა
(x_0 + passes·streamSum) mod n ცალკე periodic-reuse ექსპერიმენტია;
მისი build/multiply ხარჯიც ითვლება და ზოგად online acceleration-ად არ ცხადდება.

თითო შემდეგ ცდას წინ უძღვის: boundary/observable freeze, decode/range/alias
წესები, native C++ + Boost + უცვლელი API byte-equivalence, პატარა domain exhaustive
და corruption controls, შემდეგ assembly და paired end-to-end measurement.
დრო ცალკე დაიყოფა conversion, local work, summary/merge, normalization და
output ხარჯებად; მთავარი accept საზომი მაინც სრული region-ია.
CPU worker count observed cores-იდან headroom-ით აირჩევა; ერთი worker-ის
schedule ცალკე კონტროლი იქნება. ჯერ ახალი წარმოდგენა/კონტრაქტი, მერე გაზომილი
CPU ან SIMD პარალელიზმი — არა უბრალოდ მეტი threads.

## არტეფაქტები და აღდგენა

- [Frozen protocol](MODN_NATIVE_PROTOCOL.md), [header](probes/modn_add_control.hpp),
  [C++ tests](tests/test_modn_add_control.cpp), [C++ driver](probes/modn_add_compare.cpp).
- [წინასწარი გეგმა](data/modn_native_run_plan_20260905.json):
  2026-09-05 15:12:58 UTC; SHA256
  `6c2b0e3ff5569b3b58a6fd30d38beb435362e74ba9c8670c4ba6a6029a32b9ac`.
  მისი `measurements_started:false` რეგისტრაციის მომენტის უცვლელი snapshot-ია.
- [Validation evidence](data/modn_native_validation_20260905.json):
  ექვსი build command/result, ოთხი correctness output, რვა smoke raw output,
  უარყოფითი CLI და გარემო.
- [Raw numeric audit](data/modn_native_numeric_audit_20260905.json):
  რვავე raw ფაილის ბმულის path, სტატისტიკა და manager audit-ის executable JS source.
- [Disassembly](data/modn_native_disassembly_20260905.txt):
  8 timed wrapper, 3 throughput body, Scalar+= და add_impl.
- [Manifest](data/modn_native_manifest_20260905.json): source/binary/artifact SHA256,
  გარემოს endpoints, დამოუკიდებელი review, საზღვრები და command provenance.
  Measured binary: `/tmp/parseatlas-modn-review.7b5Pcg/modn_compare`,
  SHA256 `0ef42ba836bc7b29a567c330d6d6394fcf8ddf85c4dafe39f041082581df991c`.
  Temporary executable მუდმივი archive არაა; source/flags/hash/assembly შენახულია.

Task MCP მფლობელის პირდაპირი მითითებით დროებით გამოტოვებულია.
`PA_SECP_M1_MODN_NATIVE_015` არ გაშვებულა/მიღებულა/დახურულა ამ direct რეჟიმში;
მისი launch blocker-ის ბარათია NF-2026-00012. ეს ჩანაწერი canonical Task acceptance
არ არის. Session identity შენარჩუნებულია: `01a06be6-2904-7c62-9e7d-1245c34a5312`.
