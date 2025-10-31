import os

START = "<s>"   # symbol to mark the start of a word
END = "</s>"    # symbol to mark the end of a word

def loadPairs():
    path = os.path.join(os.getcwd(), "aspell.txt")

    pairsLower = []   # (correct_lower, typo_lower)
    caseMap = {}      # correct_lower -> first-seen "Correct"
    typoCounts = {}   # typo_lower -> {correct_lower -> count}

    file = open(path, "r", encoding="utf-8")
    for line in file:
        line = line.strip()
        if not line or ":" not in line:
            continue
        left, right = line.split(":", 1)
        correctOrig = left.strip()
        correctLow = correctOrig.lower()
        if correctLow not in caseMap:
            caseMap[correctLow] = correctOrig

        typoParts = right.strip().split()
        i = 0
        while i < len(typoParts):
            tOrig = typoParts[i].strip()
            if tOrig:
                tLow = tOrig.lower()
                pairsLower.append((correctLow, tLow))

                # count (typo -> correct) frequency
                if tLow not in typoCounts:
                    typoCounts[tLow] = {}
                if correctLow in typoCounts[tLow]:
                    typoCounts[tLow][correctLow] = typoCounts[tLow][correctLow] + 1
                else:
                    typoCounts[tLow][correctLow] = 1
            i += 1
    file.close()

    # build a single best mapping typo -> most frequent correct
    typoLex = {}  # typo_lower -> best correct_lower
    for tLow in typoCounts:
        bestC = None
        bestN = -1
        inner = typoCounts[tLow]
        for cLow in inner:
            n = inner[cLow]
            if n > bestN:
                bestN = n
                bestC = cLow
        typoLex[tLow] = bestC

    return pairsLower, caseMap, typoLex

# Extract unique letters from correct and typo words
def buildAlphabets(pairs):
    stateSet = {}   # letters appearing in correct words
    obsSet = {}     # letters appearing in typos
    for (correct, typo) in pairs:
        for c in correct:
            stateSet[c] = 1
        for o in typo:
            obsSet[o] = 1
    states = sorted(stateSet.keys())
    observations = sorted(obsSet.keys())
    return states, observations

def add1Smooth(counts, vocab):
    # Makes sure every possible outcome has a small nonzero probability.
    # Prevents zero probabilities that would kill paths in the Viterbi algorithm.
    total = 0
    for v in vocab:
        if v in counts:
            total += counts[v]
    vocabSize = len(vocab)
    probs = {}
    denom = float(total + vocabSize)  # add 1 for each possible symbol
    for v in vocab:
        if v in counts:
            probs[v] = float(counts[v] + 1) / denom
        else:
            probs[v] = 1.0 / denom
    return probs

# Helper function for adding to nested dictionaries manually
def increment1(a, b, counts):
    if a not in counts:
        counts[a] = {}
    if b in counts[a]:
        counts[a][b] = counts[a][b] + 1
    else:
        counts[a][b] = 1

def buildTransitions(pairs, states):
    # Counts how often each letter follows another in correct words
    counts = {}

    # Go through every correct word and count transitions between letters
    for (correct, _) in pairs:
        if len(correct) == 0:
            continue
        increment1(START, correct[0], counts)
        i = 0
        while i + 1 < len(correct):
            increment1(correct[i], correct[i + 1], counts)
            i += 1
        increment1(correct[-1], END, counts)

    if START not in counts:
        counts[START] = {}
    for s in states:
        if s not in counts:
            counts[s] = {}

    targets = []
    for s in states:
        targets.append(s)
    targets.append(END)

    # Convert raw counts to probabilities
    trans = {}
    trans[START] = add1Smooth(counts[START], targets)
    for s in states:
        trans[s] = add1Smooth(counts[s], targets)
    return trans

def increment2(s, o, counts):
    if s not in counts:
        counts[s] = {}
    if o in counts[s]:
        counts[s][o] = counts[s][o] + 1
    else:
        counts[s][o] = 1

def buildEmissions(pairs, states, observations):
    # Counts how often each correct letter is mistyped as another letter
    counts = {}

    for (correct, typo) in pairs:
        m = len(correct)
        if len(typo) < m:
            m = len(typo)
        i = 0
        while i < m:
            increment2(correct[i], typo[i], counts)
            i += 1

    for s in states:
        if s not in counts:
            counts[s] = {}

    emit = {}
    for s in states:
        emit[s] = add1Smooth(counts[s], observations)
    return emit


def viterbiDecode(word, states, trans, emit):
    # Reconstructs the most likely correct word given a possibly misspelled one
    if len(word) == 0:
        return ""

    # initialize with probabilities from START -> first letter
    vPrev = {}
    for s in states:
        if START in trans and s in trans[START]:
            pStart = trans[START][s]
        else:
            pStart = 0.0
        if s in emit and word[0] in emit[s]:
            pEmit = emit[s][word[0]]
        else:
            pEmit = 0.0
        vPrev[s] = pStart * pEmit

    backPointers = []

    # for each following letter pick the most likely previous letter
    t = 1
    while t < len(word):
        vCurr = {}
        bpCurr = {}
        for s in states:
            best = -1.0
            bestPrev = None
            if s in emit and word[t] in emit[s]:
                pEmit = emit[s][word[t]]
            else:
                pEmit = 0.0
            # Try every possible previous letter and keep the best one
            for sp in states:
                if sp in vPrev:
                    pPrev = vPrev[sp]
                else:
                    pPrev = 0.0
                if sp in trans and s in trans[sp]:
                    pTrans = trans[sp][s]
                else:
                    pTrans = 0.0
                score = pPrev * pTrans * pEmit
                if score > best:
                    best = score
                    bestPrev = sp
            vCurr[s] = best
            bpCurr[s] = bestPrev
        backPointers.append(bpCurr)
        vPrev = vCurr
        t += 1

    # pick the most likely final letter
    bestFinal = -1.0
    lastState = None
    for s in states:
        if s in vPrev:
            pLast = vPrev[s]
        else:
            pLast = 0.0
        if s in trans and END in trans[s]:
            pEnd = trans[s][END]
        else:
            pEnd = 0.0
        score = pLast * pEnd
        if score > bestFinal:
            bestFinal = score
            lastState = s

    if lastState is None:
        return word

    # Backtrack using stored best choices to reconstruct correct letters
    path = [lastState]
    i = len(word) - 2
    while i >= 0:
        bp = backPointers[i]
        if path[-1] in bp:
            prevState = bp[path[-1]]
        else:
            prevState = None
        if prevState is None:
            break
        path.append(prevState)
        i -= 1
    path.reverse()

    return "".join(path)

def main():
    pairs, caseMap, typoLex = loadPairs()
    states, observations = buildAlphabets(pairs)
    trans = buildTransitions(pairs, states)
    emit = buildEmissions(pairs, states, observations)

    line = input("Type text to correct: ").strip()
    words = line.split()

    output = []
    i = 0
    while i < len(words):
        w = words[i]
        wLow = w.lower()

        # if present in aspell.txt
        if wLow in typoLex:
            decodedLow = typoLex[wLow]
        else:
            # otherwise, use HMM/Viterbi
            decodedLow = viterbiDecode(wLow, states, trans, emit)

        restored = decodedLow
        if decodedLow in caseMap:
            restored = caseMap[decodedLow]

        output.append(restored)
        i += 1

    print("Corrected:", " ".join(output))

if __name__ == "__main__":
    main()
