# L0 — სად შეიძლება გადავიტანოთ დამოკიდებული ჯაჭვის მოგება

თარიღი: 2026-09-05. სტატუსი: **SOURCE-MAPPED, performance transfer unmeasured**.
ბრენჩი: `experiment/representation-search`.
შესწავლილი production რევიზია: `fef231d4e4173bd016fb2a3a1eff67087396a203`.

## მოკლე პასუხი

ყველაზე ახლო საცდელი ადგილებია **4×64 scalar-ის mod-n შეკრება** და
**4×64 FieldElement-ის mod-p შეკრება**, როცა მათი შედეგი შემდეგ ოპერაციას
ბლოკავს. ხელმოწერასაც აქვს შესაბამისი ოთხლიმბიანი შეკრება, მაგრამ იგი
ცალკე **CT scalar** გზაზეა და დამატებით უსაფრთხოების gates მოითხოვს.
არცერთ ამ ადგილას 1.5× მოგება ჯერ არ გაზომილა.

[წინა native შედეგი](L0_NATIVE_COMPARISON.md) იყო კონკრეტული rotate/XOR-ით
განახლებადი ჯაჭვის წყვილური მედიანა **1.497× / 1.534×**. მისი დროში შედიოდა
conditioning, state-ის შეფუთვა და compiler-ის register/spill გადაწყვეტილებები.
დამოუკიდებელ პატარა და დიდ მასივებზე Original უფრო სწრაფი იყო. ამიტომ
ეს ჯერ არ ადგენს ზოგადად უფრო სწრაფ adder-ს ან ყოველი `add64`-ის ჩანაცვლების
ტექნიკურ სარგებელს. State-ის შეფუთვის შემდგომი კონტროლი ცალკე ანგარიშშია.

მფლობელმა მკაფიოდ დაადასტურა: **რეალურ გამოყენებაში გამარჯვებული მეთოდით ძველი
კოდი ჩანაცვლდება**. გაყინულია შედარების რეფერენსი და არა სამუშაო ბრენჩი.
ქვემოთ მოცემული gates არის შერჩევის საზომები, არა ცვლილების აკრძალვა.

## კონკრეტული წყაროს რუკა

ქვემოთ პრიორიტეტი ნიშნავს **შემდეგი კონტროლირებადი ექსპერიმენტის რიგს** და
არა უკვე დადასტურებულ სარგებელს. Source Graph-ის inferred call edges-ს
დაემატა ქვემოთ მითითებული ზუსტი bodies-ის კითხვა; runtime სიხშირე არ გაზომილა.

| რიგი | კონკრეტული ადგილი და პირდაპირი კავშირი | რა შეიძლება გამოვცადოთ | რატომ არ გადადის 1.5× ავტომატურად |
|---|---|---|---|
| 1 | [scalar.cpp:80](../../src/cpu/src/scalar.cpp#L80) `add_impl`; მას იძახებს `fast::Scalar::operator+` (334) და `operator+=` (598) | უცვლელი mod-n reduction-ის წინ ოთხი `add64`-ის ჯაჭვის ალტერნატივა | რეალური cin=0; ემატება `order_overflow`, branch და საჭიროებისას ORDER-ის გამოკლება; წინა recurrence განსხვავებულია |
| 2 | [field.cpp:422](../../src/cpu/src/field.cpp#L422) generic `add_impl`; მას პირდაპირ იძახებს `FieldElement::operator+` (3524) და `operator+=` (3556) | `a+b` და შემდეგ `s+C`, C=2^32+977, როგორც მთელი mod-p ოპერაცია; თითო ჯაჭვის ცალკე ჩანაცვლებაც კონტროლია | საჭიროა ორივე carry, reduction-ის არჩევა და canonical output; constant addend-ზე compiler-ის schedule განსხვავდება |
| 3, ცალკე უსაფრთხოების გზა | [ct_scalar.cpp:26](../../src/cpu/src/ct_scalar.cpp#L26) `add256_scalar` → `ct::scalar_add` (65) | იმავე CT implementation-ის წინააღმდეგ ოთხლიმბიანი carry-ის ახალი schedule, სრული CT reduction-ით | მიმდინარე CT baseline იყენებს `add_carry_u64`-ს, არა წინა Original-ის `detail::add64`-ს; barriers/masks/select და compiled leakage contract უნდა შენარჩუნდეს |
| 4 | [scalar.cpp:105](../../src/cpu/src/scalar.cpp#L105) და [field.cpp:404](../../src/cpu/src/field.cpp#L404) `sub_impl`-ის correction-add | borrow-ის შემდეგ modulus-ის დამატების ოთხლიმბიანი მონაკვეთი | გამოკლება არ გამოგვიცდია; ეს მხოლოდ correction-add-ის კანდიდატია და მთელი subtract-ის ახალი baseline სჭირდება |

Generic FieldElement add/sub ირჩევა `SECP256K1_PLATFORM_STM32 &&
(__arm__ || __thumb__)` გზის `#else`-ში ([field.cpp:252](../../src/cpu/src/field.cpp#L252),
400–449). STM32/ARM გზა რვა 32-ბიტიან სიტყვასა და `arm_add256`-ს იყენებს;
მასზე ოთხი 64-ბიტიანი limb-ის გაზომვა ვერ გადაიტანება.
შესწავლილ x86 generic `FieldElement::operator+`/`+=` call sites-ში assembly
add-ზე ცალკე dispatch არ არის: ორივე პირდაპირ C++ `add_impl`-ს იძახებს.
ეს დასკვნა არ ვრცელდება გამრავლების ან ყველა point backend-ის dispatch-ზე.

## ხელმოწერის რეალური გზა და მნიშვნელოვანი გამორიცხვა

ECDSA-ს წყაროში [ecdsa.cpp:700](../../src/cpu/src/ecdsa.cpp#L700) და
[ecdsa.cpp:760](../../src/cpu/src/ecdsa.cpp#L760) იყენებს
`ct::scalar_mul(k_inv, ct::scalar_add(z, ct::scalar_mul(r, private_key)))`-ს.
Schnorr-ის [schnorr.cpp:486](../../src/cpu/src/schnorr.cpp#L486) იყენებს
`ct::scalar_add(k, ct::scalar_mul(e, kp.d))`-ს.
ეს არის კონკრეტული, წყაროთი დადასტურებული გამოყენების შესაძლებლობა —
**არა** მტკიცება, რომ `fast::Scalar::operator+`-ის შეცვლა signing-ს ააჩქარებს.

`ct::scalar_add`-ში სრული გზა მოიცავს `add256_scalar`, modulus-ის გამოკლებას,
`value_barrier(carry/borrow)`, masks-ს, `cmov256`-სა და `Scalar::from_limbs`-ს.
Candidate მხოლოდ ამ გარე contract-ის შენარჩუნებით განიხილება. Fast გზით
ჩანაცვლება, secret-dependent branch-ის დამატება ან reduction-ის მოცილება
დაუშვებელია. Source-ში branchless ფორმა და finite byte equality თავისთავად
compiled constant-time ან leakage მტკიცებულება არ არის.
ამ CT helper-ის performance, მისი carry helper-ის ქვედა backend და signing-ის
დროითი წილი ამ რუკის ფარგლებში არ გაზომილა.

## რატომ არ არის ეს პირდაპირ point arithmetic-ის 1.5× აჩქარება

[config.hpp:53](../../src/cpu/include/secp256k1/config.hpp#L53) განსაზღვრავს
`SECP256K1_FE52_COMPUTE`-ის ხელმისაწვდომობას. `SECP256K1_FAST_52BIT`
storage-ის guard ცალკეა [point.hpp:23](../../src/cpu/include/secp256k1/point.hpp#L23).
მის ქვეშ `Point`-ის x/y/z არის `FieldElement52`
([point.hpp:376](../../src/cpu/include/secp256k1/point.hpp#L376)); `#else`-ში —
4×64 `FieldElement`. [point.cpp:350](../../src/cpu/src/point.cpp#L350)-ის FE52
compute განყოფილებაშიც ცალკეა native-int128 storage guard.

ნამდვილი FE52 addition body, და არა მხოლოდ კომენტარი,
[field_52_impl.hpp:2495](../../src/cpu/include/secp256k1/field_52_impl.hpp#L2495)
და `add_assign` (2506), ასრულებს **ხუთ დამოუკიდებელ limb addition-ს, carry-ის
გადატანის გარეშე**. ეს lazy წარმოდგენაა საკუთარი magnitude/normalization
contract-ით ([field_52.hpp:96](../../src/cpu/include/secp256k1/field_52.hpp#L96)).
იქ ოთხლიმბიანი GP carry-ის ჩასმა სხვა ოპერაციას დაამატებდა ან contract-ს შეცვლიდა.

ამრიგად, 4×64 mod-p adder-ის მოგებაც რომ დადასტურდეს, FE52 point გზაზე მისი
ავტომატური გადატანა არ ხდება. რეალური გამოყენების ადგილის ასარჩევად საჭიროა
მიზნობრივი build-ის guards/linked code და call-path profile. ეს რუკა source-ის
პირობით მარშრუტებს ადგენს; ყველა რეალურ აპლიკაციაში აქტიურ backend-ს არ აცხადებს.

მენეჯერის დამატებითი compile-only შემოწმება მიმდინარე GCC native flags-ით
(`-include secp256k1/config.hpp -include secp256k1/point.hpp -dM -E`) აჩვენებს
`SECP256K1_FE52_COMPUTE=1`, `SECP256K1_FAST_52BIT=1`, `__x86_64__=1`
და `__SIZEOF_INT128__=16`; `NO_INT128` და `USE_4X64_POINT_OPS` არ განისაზღვრა.
ეს header-ების შერჩევის რეალური შემოწმებაა, არა production binary-ის runtime profile.

## მიმზიდველი, მაგრამ ჯერ შეუსაბამო სამიზნეები

- `field.cpp:134`-ის `uint320_add_assign` **ხუთლიმბიანია**. არსებული ოთხლიმბიანი
  ტესტით მეხუთე სიტყვა და მისი overflow ვერ გადაიყრება. უფრო მეტიც,
  `FieldElement::inverse` (3540–3553) ამჟამინდელი source dispatch-ით იძახებს
  safegcd-ს native-int128-ზე, სხვაგვარად safegcd30-ს. UInt320/EEA helper-ის
  არსებობა არ ამტკიცებს, რომ იგი inverse-ის მიმდინარე hot path-ია.
- `scalar.cpp:494–567`-ის product/Barrett carry loops მუშაობს 5/8/9-სიტყვიან
  აკუმულატორებზე და `Scalar::operator*`-ის `SECP256K1_NO_INT128` fallback-შია.
  Native გზა (343-დან) იყენებს სხვა, მათ შორის `{c0,c1,c2}` ფართო state-ს.
  სრული 256-bit add-ის byte equality არ ამტკიცებს product-accumulator-ის
  range/overflow contract-ს.
- `field.cpp:223`-ის `add_into` მუშაობს განსხვავებული სიგრძის მასივსა და
  ცვლად საწყის ინდექსზე; `mul_wide` (468-დან) ამჟამინდელ source-ში portable
  accumulation გზაა. ამ რუკამ მისი ყველა production dispatch არ გაიარა.
- არცერთი 64-bit `detail::add64` call დამოუკიდებლად არ არის ოთხლიმბიანი kernel.
  მისი ბრმა ჩანაცვლება შეცვლის API-ის სიგანეს, carry-ის ცხოვრებასა და caller-ის
  invariants-ს. ასევე არ გამოგვიცდია GPU/SIMD/32-bit performance გადატანა.

## შემდეგი C++ ექსპერიმენტების კონტრაქტი

1. ჯერ aggregate/scalar-local state კონტროლი და compiled code review:
   გამოეყოს, რამდენად ინარჩუნებს მიმდინარე დაკვირვება უპირატესობას სხვა
   შეფუთვაში. შეცვლილი schedule-ის შედეგი მარტო spills-ის მიზეზობრივ წილს
   მაინც ვერ ამტკიცებს. უარყოფითი შედეგიც უნდა დარჩეს.
2. შემდეგ ცალკე mod-n და mod-p native experiments, გაყინული რეფერენსითა და
   იზოლირებული candidate build-ით; მოგების დადასტურებისას სამუშაო კოდი იცვლება.
   Candidate-ის სრული output შეედაროს მიმდინარე ზუსტ API baseline-სა და C++
   bigint ორაკულს: canonical 32 byte შედეგი, საჭირო raw low+carry შიდა gates.
   Boundary corpus-ში განზრახ შევიდეს 0, 1, modulus−1, modulus−2,
   `(modulus−1)+1`, `(modulus−1)+(modulus−1)`, ყველა reduction/carry რეჟიმი და
   `x += x` alias შემთხვევა. Random input მარტო იშვიათ reduction-ს ვერ ფარავს.
3. იზომოს როგორც დამოკიდებული modular chain, ისე დამოუკიდებელი throughput;
   თითო paired sample-ში ერთი და იგივე reduction/serialization boundary,
   operand corpus, სამუშაოს რაოდენობა და რეალური compiled calls. წინა raw
   kernel-ის საუკეთესო ns ამ ახალი API-ის denominator არ არის.
4. CT candidate-ს ჰქონდეს საკუთარი უცვლელი CT baseline, masks/barriers-ის
   preservation, target compiler/backend-ის assembly და leakage gates.
   ჩვეულებრივი functional benchmark CT მიღების საფუძველი არ არის.
5. მხოლოდ ამის შემდეგ შეფასდეს producer→consumer რეალური call-path, შემდეგ
   სრული operation: მაგალითად შესაბამისი signing path ან სხვა აღმოჩენილი
   caller. რეალურ გამოყენებაში გაზომვითა და შესაბამისი სისწორის/უსაფრთხოების
   gates-ით გამარჯვებული კანდიდატი სამუშაო ბრენჩში ჩაანაცვლებს ძველს;
   მხოლოდ წყაროს რუკა performance-ის მიღების მტკიცებულება არ არის.

თუ ჰიპოთეტურად ერთი ნაწილის აჩქარებაა 1.5× და იგი სრული სერიული დროის წილ f-ს
იკავებს, საბაზისო შეფასებაა `1 / ((1−f) + f/1.5)`. მაგალითად, **მხოლოდ
საილუსტრაციოდ**, f=10% იძლევა ≈1.034×-ს, f=30% — ≈1.111×-ს და არა 1.5×-ს.
ჩვენ f არ გაგვიზომავს; recurrence-ის შეფარდება ამ ფორმულაში რეალური
production ნაწილის გაზომვის ნაცვლად ვერ ჩაისმება.

## მტკიცებულების საზღვარი და ჰეშები

წყაროს რუკის ავტორს არ შეუცვლია production კოდი და არ გაუშვია build, benchmark
ან Python; მენეჯერის ცალკე preprocessor შემოწმება ზემოთაა აღწერილი. გამოიყენებოდა მენეჯერის verified Source Graph-ით ნაპოვნი ზუსტი
წყაროები და bounded reads. Worker-ის role-specific MCP tools ამ სესიაში
ხელმისაწვდომი არ იყო; ეს დაბრკოლება მენეჯერს მიეწოდა, manager role არ ყოფილა
იმიტირებული. Owner-ის მითითებით Task MCP lifecycle კვლავ შეჩერებული იყო.
ამ source-map-ის მიღება მხოლოდ independent review-ის შემდეგ ხდება.

| წყარო | SHA-256 |
|---|---|
| `src/cpu/src/scalar.cpp` | `9da20c8fbe95c531b73fd494ff8e7a969d055ae31e8a7b4b588d0885a960050e` |
| `src/cpu/src/field.cpp` | `db5792d9bc0b51cfd12927af11fae7160c1a671b8028e0c4d73ef7f3fb6e6812` |
| `src/cpu/src/ct_scalar.cpp` | `a7cc33a850084856624ab8a5c90a8048746d833fe0a643d6c1541ee6359179a9` |
| `src/cpu/include/secp256k1/detail/arith64.hpp` | `cc95a42137df602eaee3a560993b8c59c091e017364a0345be486ba012de205d` |
| `src/cpu/include/secp256k1/config.hpp` | `fd5596e13d41f7ec566af10862a31d30acde06b73570cae67f8a23540cc6b956` |
| `src/cpu/include/secp256k1/field_52.hpp` | `95ca595490b827c6cc1965a7ae2feb1ae72546c3b60dba0562f94f842f998898` |
| `src/cpu/include/secp256k1/field_52_impl.hpp` | `2c3f77d7855e9881ebeaf2b7f53de61e683de5201c77d23caaa288df8bf2afb3` |
| `src/cpu/include/secp256k1/point.hpp` | `61c35951040476ca8b0b96807dd12af2dc3a8f7b3f3f1f9f83ddfcea9ec4b2ec` |
| `src/cpu/src/point.cpp` | `1bbbac53248572ee52df6c1b84cd40529bbf676360dee8fb2df60c245c3726f0` |
| `src/cpu/src/ecdsa.cpp` | `25e445afedc8b77e69015c1da1bf452c88bbbbf5ce3bbfb43a4c302bb80fda07` |
| `src/cpu/src/schnorr.cpp` | `c6d66ed1a096079c674cb336142cca446d46518ef9adfdfecda9af9a6a29e2d5` |
| `L0_NATIVE_COMPARISON.md` | `40434a38a7d1e346d91514d1b73c7311530d1980f662500feaabfa660beaa167` |

ეს არის [25 ლინზის](L0_25_LENSES.md) V06/V08/V09/V15/V21–V24-ისთვის
დამატებითი call-path/backend/contract რუკა, არა ძველი hash-bound რეესტრის
სტატუსების ჩუმი განახლება და არა 25×25 კვლევის დასრულება.
