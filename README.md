# Párhuzamos-eszközök-programozása

Féléves feladat:

Numerikus integrálás elvégzése szekvenciálisan és OpenCL-lel való párhuzamosítással.

Integrálási módszerek:

- Simpson Rule, Left Rectangle Rule, Trapezoidal Rule 

Integrálható függvények:

- Sin(x), Cos(x), Exp(x), Sqrt(x), Log(x)

A helyes eredmény ellenőrizhető:

www.emathhelp.net/en/calculators/calculus-2/simpsons-rule-calculator/

A vizsgált sin(x) függvény esetén, a = 2, b = 10, n∈[10; 20.000.000] intervallumon a következők figyelhetőek meg:

- A számítási idő a szekvenciális futás esetén gyorsabb, mint a párhuzamosított, ha a vizsgált részintervallum mérete viszonylag kicsi (1000-es értéknél még igaz)
- Kis részintervallum vizsgálata esetén a közelítés triviálisan rendkívül pontatlan, főleg téglalap-módszer esetén
- Ahogy növeljük a vizsgált részintervallum méretét, úgy nő a pontosság is, viszont a szekvenciális futás számára az időt tekintetbe véve egyre költségesebb
- Nagyobb intervallumon vizsgálódva a szekvenciális program kritikusan lassabb, mint a párhuzamos, [10; 20.000.000] esetén nagyjából a 92-szerese időt jelenti a számítás elvégzése
