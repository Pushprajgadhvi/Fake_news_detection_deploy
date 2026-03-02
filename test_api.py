import urllib.request, json

def test(label, text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:5000/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    d = json.loads(urllib.request.urlopen(req).read())
    print(f"[{label}]")
    print(f"  Overall    : {d['overall']}")
    print(f"  Uncertain  : {d.get('uncertain')}")
    print(f"  Avg conf   : {d.get('avg_conf')}%")
    print(f"  Legit score: {d.get('legit_score')}")
    for m, v in d["models"].items():
        print(f"    {m}: {v['label']} ({v['confidence']}%)")
    if d.get("fake_reasons"):
        print("  Fake Reasons:")
        for r in d["fake_reasons"]:
            print(f"    - {r[:90]}...")
    print()

# ── Real US political news ──────────────────────────────────────────
test("US Real News",
    "The head of a conservative committee opposed to Obamacare is urging Republicans in Congress "
    "to vote for the new healthcare bill. President Donald Trump said the legislation is the best "
    "deal Congress can get on healthcare reform. Senate Majority Leader Mitch McConnell scheduled "
    "a vote for Thursday. The Congressional Budget Office scored the bill as reducing the deficit.")

# ── Real Indian news (should be Uncertain, not Fake) ────────────────
test("ISRO Real News",
    "The Indian Space Research Organisation ISRO successfully launched the Aditya-L1 mission, "
    "India first solar observatory. The launch took place at the Satish Dhawan Space Centre in "
    "Sriharikota. According to ISRO, Aditya-L1 will travel to the Lagrange Point of the Sun-Earth "
    "system where it will observe the Sun continuously. The Ministry of Science confirmed the mission.")

# ── Real Indian political news ──────────────────────────────────────
test("Indian Parliament News",
    "Prime Minister Narendra Modi announced in Lok Sabha on Monday that the government plans to "
    "introduce a new digital infrastructure bill. The Minister of Electronics stated that the bill "
    "has been approved by the Cabinet and will be tabled in the Rajya Sabha next week. "
    "According to ANI, the bill aims to regulate data privacy and strengthen cybersecurity laws.")

# ── Conspiracy fake news ────────────────────────────────────────────
test("Fake: Bleach COVID Cure",
    "BREAKING: Scientists discover that drinking bleach cures COVID-19! Doctors are hiding this "
    "from the public. The government is suppressing a breakthrough cure. Big pharma does not want "
    "you to know this secret remedy! Share this before it gets deleted! Wake up sheeple!")

# ── Political disinformation ────────────────────────────────────────
test("Fake: Soros Paid Protestors",
    "BREAKING: George Soros paid protestors 80 dollars an hour to riot in the streets and destroy "
    "property across America. Leaked Soros documents show how the billionaire globalist funded "
    "anti-Trump protests in all 50 states. Paid protestors were bussed in from out of state to "
    "cause chaos and division in American cities.")
