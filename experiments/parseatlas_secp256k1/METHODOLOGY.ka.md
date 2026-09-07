# ნორმატიული მეთოდოლოგია (ქართული)

ამ დოკუმენტსა და `METHODOLOGY.en.md`-ს თანაბარი ნორმატიული ძალა აქვს. JSON Schema
არის ID-ებისა და შესრულებადი ჩანაწერის ვალდებულებების ერთადერთი წყარო.

## ნორმატიული თანხვედრის ბლოკი

`CONTRACT_VERSION=2; WORLD_COUNT=25; VIEW_COUNT=25; MATRIX_CELLS=625; PRIMITIVE_CLASSES=4; RANKING=search_reachable>boundary_equivalent>invariants_eligible>resource_feasible>cost_pareto_within_feasible_slice; CLAIM_CLASSES=OBSERVED,PROVEN,VERIFIED,HYPOTHESIS,MEASURED,CORRECTION; HISTORICAL_REVISION=d71b406c95141d81749671431a4c0f4605e0c4e4; GUIDANCE_REVISION=13fac0a300cf07eac9424434ec1e2c6bd5f75564; PARALLEL_WORLDS_REVISION=f428fb53948bc15c4668c303366231817a947708; INPUT_EXPERIMENTS=0037,0038,0039,0040,0041; E0039=46:1_MODEL_DEFECT+8_OPERATOR_LANGUAGE+18_TOKEN_WORD+19_RULE_LIMIT; E0041=58_ENTRIES,25_WORLDS,1450_CELLS,52_FORCED,6_MOVABLE,TAX_FLOOR_21,VARIABLE_1..4,TIE_CHANGES_52,MAX_DEPTH_RATIO_BOUND_2.333,MUTANTS_19_OF_22`

## ფიქსირებული ამოცანა, ობიექტები და W01–W25

აქ გადმოტანილი დამფუძნებელი კითხვაა: წარმოდგენის სხვა საფუძვლიანი კონვენციების
მქონე მათემატიკური ტრადიციები როგორ ამოხსნიდნენ იმავე secp256k1 ამოცანას საკუთარი
ბუნებრივი პრიმიტივებით? ფიქსირებული ობიექტებია `p` მოდულის ველის ელემენტი,
განსხვავებული `n` მოდულის სკალარი, ჯგუფის წერტილი (უსასრულობის ჩათვლით) და
ხელმოწერის შემოწმების შესატანი/შედეგი. `p`-სა და `n`-ის მტკიცებულება არ იცვლება.

W01–W25 არის 25 ხელახლა გამოყენებადი, შიგნიდან თანმიმდევრული წარმოდგენის policy;
არა 25 precedence grammar, დაუკავშირებელი knob, algorithm ან აღმოჩენა. ყოველი policy
მართავს დასაშვებ derivation-ს, carried state-ს, normal form-ს, encode/decode-ს,
invariant-სა და შეფასებას. იგი ცალ-ცალკე გამოიყენება ყოველ ფიქსირებულ primitive/
region-ზე. ყოველ world/primitive უჯრაში წერია `APPLICABLE` და კონკრეტული output
representation, ან დასაბუთებული `N/A`. უცნობი N/A არ არის. collapsed/duplicate
world ინახება. 25-policy design არც სისრულეა და არც თეორემა. W01 production control-ია.

ყოველი applicable candidate-ის ვალდებულებაა
`decode_out(candidate(encode_in(x))) = baseline(x)` გამოცხადებულ input/output
domain-ზე. იწერება range, definedness/precondition, მთელი carried state,
caller-visible behavior, overflow და aliasing. boundary witness ამტკიცებს
reachability-ს და არა equivalence-ს. rational cancellation limb machine arithmetic-ს
ვერ დაამოწმებს range/overflow/finite-field ვალდებულებების გარეშე.

საწყისი მიმდინარე engine, source commit-ზე
`fef231d4e4173bd016fb2a3a1eff67087396a203`, გაყინული differential reference-ია.
ყოველი რეალური experiment იწერს source/build hash-ებს, compiler-ს, flags-სა და target-ს
და იდენტურ deterministic input-ზე ადარებს კანონიკურ output byte-ებსა და განსაზღვრულ,
caller-visible status/state-ს. ამასთან, `p`/`n` მოდულები დამოუკიდებლად მოწმდება
შეუზღუდავი სიგრძის integer-ებით. შიდა representation შეიძლება განსხვავდებოდეს,
მაგრამ დამოუკიდებლად სწორი decode/normalization ვერ დამალავს carry/range/alias
დარღვევას. `FieldElement::to_bytes` მხოლოდ შენახულ limb-ებს ასერიალიზებს
(`field.cpp:2517-2526`) და normalization არ არის. Byte corpus-ის თანხვედრა OBSERVED-ია,
არა full-domain proof. C++ reference-ის რეალური ინტეგრაცია `ORACLE_011`-ს ეკუთვნის
და ამ contract card-ის ფარგლებს სცდება.

## V01–V25 და მტკიცებულება

## კანონიკური world კატალოგი (ID სქემაზე მიუთითებს)

ყოველი მწკრივი აღწერს policy/state-ს; boundary-ს; invariant-ს; applicability-ს. W01-ის
ოთხივე primitive-ზე control-ის გარდა, support ყოველ primitive-ზე განისაზღვრება
კონკრეტული output representation-ის `APPLICABLE` ან დასაბუთებული `N/A` ჩანაწერით.

|ID|Policy და carried state|Boundary; invariant|Support|
|--|--|--|--|
|W01|production-ის უცვლელი control და state|იგივეობის boundary; baseline ქცევა უცვლელია|ოთხივე primitive|
|W02|უმცირესი არაუარყოფითი კანონიკური ნაშთი|კანონიკურ integer-ში გადასვლა; კონგრუენტულობა და range|ჩანაწერის დროს|
|W03|ცენტრირებული ნიშნიანი ნაშთი|კანონიკურ ნაშთთან ასახვა; კონგრუენტულობა და ნიშნიანი range|ჩანაწერის დროს|
|W04|შეზღუდულად redundant მნიშვნელობა და magnitude|normalize/decode boundary; კონგრუენტულობა და magnitude bound|ჩანაწერის დროს|
|W05|Montgomery-ის `xR` მდგომარეობა|R-ით encode და R-inverse decode; Montgomery congruence და range|ჩანაწერის დროს|
|W06|Barrett-ის ნაშთი და reciprocal state|კანონიკური boundary; კონგრუენტულობა და correction bound|ჩანაწერის დროს|
|W07|pseudo-Mersenne კოეფიციენტები|კოეფიციენტების pack/unpack; modulus relation და bounds|ჩანაწერის დროს|
|W08|სიტყვის სიგანის saturated limb-ების vector|limb pack boundary; radix value და carry bounds|ჩანაწერის დროს|
|W09|თავისუფალი ბიტების მქონე limb-ები და magnitude|pack/normalize boundary; radix value და limb bounds|ჩანაწერის დროს|
|W10|არაერთგვაროვანი mixed-radix digits|წონიანი დაშლა/შეკრება; value და digit bounds|ჩანაწერის დროს|
|W11|carry-save sum და carry არხები|არხების გაერთიანება; combined value და channel bounds|ჩანაწერის დროს|
|W12|CRT ნაშთების tuple|არხებად გაყოფა და reconstruction; congruence და product range|ჩანაწერის დროს|
|W13|შეზღუდული ნიშნიანი digits|signed-radix reconstruction; value და digit bounds|ჩანაწერის დროს|
|W14|ფაქტორებად შენახული product DAG|ფაქტორების evaluate boundary; definedness და product equivalence|ჩანაწერის დროს|
|W15|numerator/denominator წყვილი|დამტკიცებული inversion boundary; denominator nonzero და equivalence|ჩანაწერის დროს|
|W16|საერთო მნიშვნელობების გაზიარებული DAG|DAG-ის evaluate boundary; node definitions და output equivalence|ჩანაწერის დროს|
|W17|დაბალანსებული expression DAG|ხის evaluate boundary; associativity domain და operand multiset|ჩანაწერის დროს|
|W18|მიმდევრული accumulator state|fold-ის evaluate boundary; associativity domain და operand order|ჩანაწერის დროს|
|W19|ჯერ შეუმცირებელი accumulator და bound|გამოცხადებულ cut-ზე final reduction; growth bound და congruence|ჩანაწერის დროს|
|W20|ყოველ ნაბიჯზე eager canonical residues|კანონიკური boundary; range და congruence|ჩანაწერის დროს|
|W21|ორი დამოუკიდებელი carry chain და merge|არხების merge boundary; partition identity და carry bounds|ჩანაწერის დროს|
|W22|SIMD lane-ების tuple|lane-ების pack/unpack; lane independence და equivalence|ჩანაწერის დროს|
|W23|bitsliced bit-plane მდგომარეობა|bit transpose boundary; reconstruction და lane independence|ჩანაწერის დროს|
|W24|base და authenticated წინასწარი table|table-ის build/validate boundary; derivation და selection behavior|ჩანაწერის დროს|
|W25|მთელ region-ზე ტიპიზებული policy composition|encode/decode boundary; component და caller-visible invariants|ჩანაწერის დროს|

## კანონიკური view კატალოგი (ID სქემაზე მიუთითებს)

|ID|Observable|Measurement/assessment method|Applicability|
|--|--|--|--|
|V01|search reachability|derivation witness-ის შესრულება|ყველა primitive|
|V02|sampled agreement|seeded differential corpus; მხოლოდ OBSERVED|applicable cell|
|V03|full-domain equivalence|proof ან mechanical full-domain artifact|applicable cell|
|V04|definedness/precondition|proof და invalid-input fixture|applicable cell|
|V05|range/overflow|range proof და boundary fixture|arithmetic cell|
|V06|aliasing|alias matrix test|implementation cell|
|V07|caller-visible behavior|API differential test|group/signature; exposed field/scalar|
|V08|invariant eligibility|artifact ყოველ invariant-ზე|applicable cell|
|V09|constant-time eligibility|static trace review და leakage experiment|secret-bearing cell|
|V10|resource feasibility|გამოცხადებული hard ceiling|implementation cell|
|V11|property manifestness|preregistered syntactic extractor|declared property|
|V12|dependency depth|DAG longest-path static model|operation region|
|V13|peak live values|liveness analysis|operation region|
|V14|operation multiset|typed static counter|operation region|
|V15|conversion cost|ტიპიზებული static operation count ან გაზომილი conversion timing|non-identity boundary|
|V16|region latency|warmed repeated benchmark|executable region|
|V17|region throughput|warmed batched benchmark|executable region|
|V18|instruction count|versioned hardware counter|supported target|
|V19|cache behavior|versioned hardware counter|supported target|
|V20|code size|linked-symbol accounting|compiled implementation|
|V21|portability|declared build/test matrix|implementation cell|
|V22|compiler sensitivity|fixed version/flag matrix|compiled implementation|
|V23|policy/tie sensitivity|preregistered optimistic/pessimistic rerun|multi-derivation cell|
|V24|rediscovery/transfer|blind generator და unchanged held-out axis|generative policy|
|V25|prior-art status|reproduction/gain/novelty-review classification|positive claim|

V01–V25-ის observable, method და applicability სქემაში დამოუკიდებლადაა მოცემული.
Finite seeded differential agreement არის OBSERVED (V02), არასოდეს full-domain
equivalence (V03). PROVEN-ს სჭირდება derivation და assumptions; VERIFIED-ს —
mechanical full-domain artifact; HYPOTHESIS დაუდასტურებელია; MEASURED-ს — environment,
software versions, corpus, warm-up, სულ მცირე 2 repetitions, statistic, uncertainty,
unit, value და artifacts; CORRECTION-ს — შენარჩუნებული predecessor, reason, author და
timestamp. კლასი ჩუმად არ მაღლდება. უცნობი measurement ზუსტად `PENDING IMPORT`-ია.

უარყოფითი შედეგი ინარჩუნებს claim identity-ს, seed-ს, artifact-ს, provenance-სა და
limits-ს: unreachable, non-equivalent, invariant failure, resource infeasible,
null effect, measurement failure და collapsed duplicate. correction ისტორიას ემატება
და არ გადაწერს. invalid-state და rejection-branch fixture სავალდებულოა.

## Reachability-first T31 ranking

რანჟირება ლექსიკოგრაფიულია: search reachability, proved boundary equivalence,
invariant/constant-time eligibility და resource feasibility. მხოლოდ გამოცხადებულ
feasible slice-ში შედარდება eligible world-ის გაზომილი cost. dependency depth,
peak live values, operation multiset, conversion cost, region latency და throughput
ცალკე Pareto coordinate-ებია. static depth და unit-cost model measurement არ არის.
property manifestness equivalence-ისა და cost-ისგან ცალკეა.

Generator scope აცხადებს ნებადართულ token-word/state ცვლილებებს და ინახავს derivation
witness-ს. Rediscovery control generator-ში target identity-ს არ დებს; უცვლელი axis
შემდეგ held-out object-ზე მოწმდება. ანგარიში არჩევს known-method reproduction-ს,
observed implementation gain-ს და prior-art review-ის მომლოდინე possible novelty-ს.

## შემოტანილი მტკიცებები და შეზღუდვები

უცვლელი ისტორიული წყაროა ParseAtlas Direction B
`d71b406c95141d81749671431a4c0f4605e0c4e4`, განსაკუთრებით 0037–0040. guidance
ეკუთვნის `13fac0a300cf07eac9424434ec1e2c6bd5f75564`-ს, ხოლო 0041 —
`f428fb53948bc15c4668c303366231817a947708`-ს; ახალი მტკიცება ისტორიას არ გადაწერს.
0033-ში ერთი rational deferred-division axis ხელითაა მიწოდებული; ეს არც autonomous
generation-ია და არც finite-field proof procedure. general synthesis, finite-field
discharge და ranking ამ engine-ში განხორციელებამდე/გაზომვამდე HYPOTHESIS/planned-ია.

0039-ის 46 missing pair იყოფა ზუსტად ასე: 1 model defect, 8 operator-language limit,
18-ს სხვა token word სჭირდება და 19 declared rule limit-ია. ამიტომ არც 37 residue
pair არის მთლიანად shape-rule limit და არც “no world shows everything” უნივერსალური;
ორივე corpus/rule-relative-ია.

Pinned 0041-ში 58 entry × 25 world = 1450 cell: 52 forced და 6 movable; tax floor 21,
variable remainder 1..4. optimistic/pessimistic tie ცვლის 52/1450 cell-ს. maximum
depth ratio 2.333 არის BOUND fixed-word unit-cost model-ში და არა secp runtime.
ფიქსირებული tree-ის depth უცვლელია; იცვლება მხოლოდ equivalent writing-ის არჩევით.
constant-floor rank preservation არითმეტიკული identity-ა და არა validation; bracket
tax limb performance-ზე არ გადადის. მოკვდა 19/22 mutant; აკლდა 2 rejection-branch
fixture და 1 probe redundant იყო, ამიტომ full mutation coverage არ მტკიცდება.

ყველა დასკვნა შეზღუდულია გამოცხადებული object, policy, view, corpus, rules, platform,
compiler და versions-ით. ცალკე შედეგი არ ამტკიცებს cryptographic security-ს,
portability-ს, constant time-ს, general performance-ს ან novelty-ს.
