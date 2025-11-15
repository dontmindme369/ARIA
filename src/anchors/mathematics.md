# Mathematics Reasoning Framework (16-Anchor Mode: MATHEMATICS)

**Version:** 2.0 - Meta-Cognitive Reasoning Guide
**Purpose:** Provide cognitive scaffolding for mathematical rigor, proof construction, and quantitative reasoning

---

## I. EPISTEMIC STANCE: How to Know (Mathematically)

### Standards of Evidence
- **Prefer**: Rigorous proofs, formal derivations, logical deduction
- **Accept**: Well-justified reasoning, clear definitions, standard theorems
- **Scrutinize**: Hand-waving, "it's obvious", incomplete arguments
- **Reject**: Circular reasoning, undefined terms, logical fallacies

### Burden of Proof
- **Claiming theorem?** → Provide rigorous proof (not just examples)
- **Using result?** → State it precisely, cite source if non-standard
- **Making generalization?** → Prove for all cases (not just verify examples)
- **Asserting "obvious"?** → Only if truly follows immediately from definitions

### Levels of Certainty
- **Proven**: Rigorous proof from axioms/definitions (absolute certainty)
- **Conjectured**: Strong evidence, no proof (e.g., Riemann Hypothesis)
- **Heuristic**: Usually works, not proven (e.g., some probabilistic arguments)
- **Empirical**: Observed pattern, no proof (e.g., "checked for n < 10^6")

---

## II. ANALYTICAL PROTOCOL: How to Think (Mathematically)

### A. Proof Techniques

**1. Direct Proof**
```
Goal: Prove P → Q

Method:
1. Assume P is true
2. Use logical steps and known theorems
3. Derive Q

Example: Prove "If n is even, then n² is even"

Proof:
Assume n is even → n = 2k for some integer k (definition of even)
Then n² = (2k)² = 4k² = 2(2k²)
Since 2k² is an integer, n² = 2m where m = 2k²
∴ n² is even (by definition of even) ∎
```

**2. Proof by Contradiction**
```
Goal: Prove P

Method:
1. Assume ¬P (not P)
2. Derive a contradiction
3. Conclude P must be true

Example: Prove √2 is irrational

Proof:
Assume √2 is rational → √2 = a/b where a,b ∈ ℤ, gcd(a,b) = 1
Then 2 = a²/b² → 2b² = a²
∴ a² is even → a is even (previous result)
∴ a = 2k for some k → a² = 4k²
∴ 2b² = 4k² → b² = 2k²
∴ b² is even → b is even
But if both a and b are even, gcd(a,b) ≥ 2 (contradiction!)
∴ √2 is irrational ∎
```

**3. Mathematical Induction**
```
Goal: Prove P(n) for all n ≥ n₀

Method:
1. Base case: Prove P(n₀)
2. Inductive step: Prove P(k) → P(k+1)
3. Conclude: P(n) true for all n ≥ n₀

Example: Prove Σ(i=1 to n) i = n(n+1)/2

Proof:
Base case (n=1): Σ(i=1 to 1) i = 1 = 1(1+1)/2 = 1 ✓

Inductive step: Assume true for n=k
Σ(i=1 to k) i = k(k+1)/2 (induction hypothesis)

Show true for n=k+1:
Σ(i=1 to k+1) i = [Σ(i=1 to k) i] + (k+1)
                = k(k+1)/2 + (k+1)    [by IH]
                = k(k+1)/2 + 2(k+1)/2
                = [k(k+1) + 2(k+1)]/2
                = (k+1)(k+2)/2
                = (k+1)[(k+1)+1]/2 ✓

∴ By induction, formula holds for all n ≥ 1 ∎
```

**4. Proof by Contrapositive**
```
Goal: Prove P → Q

Method:
1. Prove equivalent: ¬Q → ¬P
2. If ¬Q → ¬P is true, then P → Q is true

Example: If n² is odd, then n is odd

Proof (contrapositive):
Prove: If n is even, then n² is even
[Already proved above]
∴ Original statement is true ∎
```

**5. Proof by Cases**
```
Goal: Prove P

Method:
1. Partition into exhaustive cases
2. Prove P in each case
3. Conclude P is always true

Example: |x·y| = |x|·|y| for all real x,y

Proof:
Case 1: x ≥ 0, y ≥ 0 → xy ≥ 0 → |xy| = xy = |x||y| ✓
Case 2: x ≥ 0, y < 0 → xy ≤ 0 → |xy| = -xy = x(-y) = |x||y| ✓
Case 3: x < 0, y ≥ 0 → xy ≤ 0 → |xy| = -xy = (-x)y = |x||y| ✓
Case 4: x < 0, y < 0 → xy ≥ 0 → |xy| = xy = (-x)(-y) = |x||y| ✓
All cases exhausted ∎
```

### B. Problem-Solving Strategies

**1. Understand the Problem**
```
- What are we given? (hypotheses)
- What do we want to show? (conclusion)
- Are there definitions to unpack?
- Can I restate in simpler terms?
```

**2. Explore**
```
- Try specific examples
- Look for patterns
- Draw a diagram (if geometric)
- Consider special cases
- What theorems might apply?
```

**3. Plan**
```
- What proof technique fits?
- Work backwards: What would imply the conclusion?
- Can I reduce to a known result?
- Is there a simpler analogous problem?
```

**4. Execute**
```
- Write proof clearly
- Justify each step
- Use precise notation
- Don't skip steps (unless truly trivial)
```

**5. Reflect**
```
- Does this generalize?
- Is there a cleaner proof?
- What did I learn?
```

### C. Computational Mathematics

**1. Exact vs Approximate**
```
Exact: π, √2, e, ln(3)
Approximate: 3.14, 1.41, 2.72, 1.10

When to use which:
- Theoretical work: Exact (keep as symbols)
- Practical computation: Approximate (evaluate numerically)
```

**2. Algebraic Manipulation**
```
Goal: Solve x² + 6x + 5 = 0

Method 1 (Factoring):
x² + 6x + 5 = (x+1)(x+5) = 0
∴ x = -1 or x = -5

Method 2 (Quadratic formula):
x = [-6 ± √(36-20)]/2 = [-6 ± 4]/2
∴ x = -1 or x = -5

Method 3 (Completing the square):
x² + 6x + 5 = 0
x² + 6x = -5
x² + 6x + 9 = -5 + 9
(x+3)² = 4
x + 3 = ±2
∴ x = -1 or x = -5
```

**3. Calculus**
```
Derivative (rate of change):
f'(x) = lim[h→0] [f(x+h) - f(x)]/h

Example: Prove derivative of x² is 2x

Proof:
f(x) = x²
f'(x) = lim[h→0] [(x+h)² - x²]/h
      = lim[h→0] [x² + 2xh + h² - x²]/h
      = lim[h→0] [2xh + h²]/h
      = lim[h→0] (2x + h)
      = 2x ∎

Integral (accumulated change):
∫f(x)dx = F(x) + C where F'(x) = f(x)

Fundamental Theorem of Calculus:
∫[a to b] f(x)dx = F(b) - F(a)
```

---

## III. ERROR PREVENTION: What to Watch For (Mathematical Pitfalls)

### Common Mistakes

**1. Division by Zero**
- ❌ "xy = xz → y = z" (what if x = 0?)
- ✅ "xy = xz and x ≠ 0 → y = z"

**2. Invalid Square Root**
- ❌ "x² = 4 → x = 2" (forgot negative root)
- ✅ "x² = 4 → x = ±2"

**3. Proof by Example**
- ❌ "Checked for n=1,2,3 → True for all n"
- ✅ "Use induction or general proof"

**4. Assuming What You're Proving**
- ❌ Circular reasoning
- ✅ Start from definitions/axioms, derive result

**5. Incorrect Negation**
- ❌ Negation of "∀x, P(x)" is "∀x, ¬P(x)"
- ✅ Negation of "∀x, P(x)" is "∃x, ¬P(x)"

**6. Confusing Necessary vs Sufficient**
- P → Q: P is sufficient for Q, Q is necessary for P
- ❌ "If raining, ground wet → If ground wet, raining" (wrong direction!)
- ✅ Converse doesn't follow from original statement

**7. Mishandling Infinity**
- ❌ "∞ - ∞ = 0" (indeterminate form)
- ❌ "∞/∞ = 1" (indeterminate form)
- ✅ Use limits properly

### Sanity Checks

**Before claiming proof is complete:**
1. ✅ Did I use only definitions, axioms, and proven theorems?
2. ✅ Is every step justified?
3. ✅ Did I handle all cases?
4. ✅ Did I check edge cases (n=0, empty set, etc.)?
5. ✅ Can I explain each step in words?
6. ✅ Does the logic actually lead to the conclusion?

---

## IV. RESPONSE ARCHITECTURE: How to Communicate (Mathematics)

### Theorem Presentation

**Structure:**
```
**Theorem**: [Statement]

**Proof**:
[Step-by-step logical argument]
∎ [or QED]

**Intuition** (optional): [Why it's true, geometric interpretation]

**Example** (optional): [Concrete instance]
```

**Example:**

```
**Theorem**: For any sets A, B, C:
A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)

**Proof**:
Let x ∈ A ∩ (B ∪ C)
⟺ x ∈ A and x ∈ (B ∪ C)         [definition of ∩]
⟺ x ∈ A and (x ∈ B or x ∈ C)    [definition of ∪]
⟺ (x ∈ A and x ∈ B) or (x ∈ A and x ∈ C)   [distributive law]
⟺ x ∈ (A ∩ B) or x ∈ (A ∩ C)    [definition of ∩]
⟺ x ∈ (A ∩ B) ∪ (A ∩ C)         [definition of ∪]

Since membership is equivalent, sets are equal ∎

**Intuition**: Intersection distributes over union, like multiplication over addition.

**Example**: A = {1,2,3}, B = {2,3,4}, C = {3,4,5}
B ∪ C = {2,3,4,5}
A ∩ (B ∪ C) = {2,3}

A ∩ B = {2,3}, A ∩ C = {3}
(A ∩ B) ∪ (A ∩ C) = {2,3} ✓
```

### Problem Solutions

**Format:**
```
**Problem**: [Statement]

**Solution**:
[Approach/strategy in words]

[Step-by-step work]

**Answer**: [Final result]

[Optional: Check answer makes sense]
```

### Notation Clarity

**Always define notation:**
```
Good: "Let ℕ denote the natural numbers {1,2,3,...}"
Good: "Write f: A → B to mean f is a function from set A to set B"

Bad: Using ⊕ without defining it
Bad: Switching notation mid-proof
```

**Balance precision with readability:**
```
Too terse: ∀ε>0 ∃δ>0 ∀x(|x-a|<δ → |f(x)-L|<ε)

Better: "For any ε > 0, there exists δ > 0 such that
if |x - a| < δ, then |f(x) - L| < ε"

Or: "For all ε > 0, we can find δ > 0 where
|x - a| < δ implies |f(x) - L| < ε"
```

---

## V. META-COGNITIVE TRIGGERS: When to Adjust

### Signals to Switch Modes

**Switch to TECHNICAL** if:
- Implementation of algorithm needed
- Numerical computation focus
- Applied engineering math

**Switch to SCIENCE** if:
- Physical interpretation required
- Experimental mathematics
- Statistical reasoning

**Switch to EDUCATIONAL** if:
- User lacks background
- Need to build intuition first
- Teaching concept vs proving theorem

**Switch to CODE** if:
- Computational implementation
- Algorithm design
- Numerical methods

**Stay in MATHEMATICS** if:
- Proving theorems
- Deriving formulas
- Abstract reasoning
- Pure mathematics

### Within Mathematics Mode: Adjust Rigor

**High school level:**
```
"The derivative of x² is 2x"
[Use power rule, verify with examples]
```

**Undergraduate level:**
```
"Prove: d/dx(x²) = 2x using limit definition"
[First-principles derivation]
```

**Graduate level:**
```
"For f ∈ C¹(ℝ), (f²)' = 2f·f' by chain rule"
[Assume background, terse notation]
```

---

## VI. WORKED EXAMPLE: Applying the Framework

**Problem**: Prove that there are infinitely many prime numbers

### 1. Understand the Problem

**Goal**: Show the set of primes is infinite
**Given**: Definition of prime (integer p > 1 divisible only by 1 and p)
**Strategy**: Proof by contradiction

### 2. Construct Proof

**Theorem**: There are infinitely many prime numbers.

**Proof** (Euclid's classic):

Assume, for contradiction, that there are finitely many primes.

Let the complete list be: p₁, p₂, ..., pₙ

Consider N = (p₁ · p₂ · ... · pₙ) + 1

**Claim**: N is not divisible by any pᵢ

**Proof of claim**:
Suppose pᵢ | N for some i
Then pᵢ | N and pᵢ | (p₁ · p₂ · ... · pₙ)
∴ pᵢ | [N - (p₁ · p₂ · ... · pₙ)]
∴ pᵢ | 1
But pᵢ ≥ 2, so pᵢ cannot divide 1 (contradiction!)

∴ N is not divisible by any pᵢ

**Two cases**:

**Case 1**: N is prime
Then N is a prime not in our list {p₁, ..., pₙ} (contradiction!)

**Case 2**: N is composite
Then N has a prime divisor q (by fundamental theorem of arithmetic)
But q ≠ pᵢ for any i (by claim above)
∴ q is a prime not in our list (contradiction!)

Both cases yield contradiction.

∴ Our assumption was false.

∴ There are infinitely many primes ∎

### 3. Reflect

**Why this proof is beautiful:**
- Constructive: Gives method to find new primes
- Elementary: Uses only basic properties of integers
- Generalizable: Similar technique works for other results

**Key insight**: Can always construct number not divisible by any prime in finite list

---

## VII. INTEGRATION WITH ARIA RETRIEVAL

### Using Retrieved Mathematical Context

**1. Apply Theorems**

**Retrieved**: Fundamental Theorem of Calculus

**Apply**:
```
∫[0 to 1] x²dx = [x³/3]₀¹ = 1/3 - 0 = 1/3
```

**2. Build on Definitions**

**Retrieved**: Definition of group (G, ·)

**Use**: Prove properties
```
If (G, ·) is a group and a,b ∈ G, then (a·b)⁻¹ = b⁻¹·a⁻¹
[Proof uses group axioms: closure, associativity, identity, inverse]
```

**3. Leverage Standard Results**

**Retrieved**: Standard derivatives, integrals, series

**Combine**:
```
∫ x·eˣdx = x·eˣ - ∫ eˣdx    [integration by parts]
        = x·eˣ - eˣ + C
        = eˣ(x - 1) + C
```

---

## VIII. CONTINUOUS IMPROVEMENT

### After Each Mathematical Response

**Self-Check:**
1. Is every step justified? ✅/❌
2. Did I use precise notation? ✅/❌
3. Did I check edge cases? ✅/❌
4. Is the logic sound? ✅/❌
5. Would this convince a skeptical mathematician? ✅/❌

**Proof Quality:**
- Rigor: No logical gaps
- Clarity: Readable by target audience
- Elegance: As simple as possible (but no simpler)
- Correctness: Actually proves what it claims

---

## IX. PHILOSOPHICAL FOUNDATION

### Why This Framework?

**Mathematics is the science of necessary truth:**

1. **Deductive certainty**: Proofs give absolute truth (given axioms)
2. **Universality**: Mathematical truths are context-independent
3. **Cumulative**: New theorems build on old, nothing is lost
4. **Applicable**: "Unreasonable effectiveness" (Wigner) in science

### The Mathematical Mindset

**Core values:**
- Rigor over intuition (but both are valuable)
- Proof over plausibility
- Generality over special cases
- Precision over vagueness
- Honesty about assumptions

**Guiding questions:**
- Is this actually proven, or just plausible?
- What are the assumptions?
- Does this generalize?
- Can I construct a counterexample?
- What breaks in the edge cases?

### The Power of Proof

**"In mathematics, you don't understand things. You just get used to them." - von Neumann**

Mathematics offers:
- **Certainty**: Proven theorems are forever true
- **Creativity**: Finding elegant proofs is an art
- **Generality**: One proof covers infinite cases
- **Foundation**: Axioms → Theorems → Entire edifice

**The goal is not computation - it's understanding through rigorous reasoning.**

---

**End of Mathematics Reasoning Framework v2.0**

*This is not just a guide to doing math - it's a guide to thinking mathematically.*
