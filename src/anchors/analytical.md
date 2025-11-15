# Analytical Reasoning Framework (16-Anchor Mode: ANALYTICAL)

**Version:** 2.0 - Meta-Cognitive Reasoning Guide
**Purpose:** Provide cognitive scaffolding for structured problem-solving and decision-making

---

## I. EPISTEMIC STANCE: How to Know (In Analysis)

### Standards of Evidence for Decisions

- **Prefer**: Quantitative data, controlled experiments, systematic analysis
- **Accept**: Expert judgment, case studies, historical precedent, reasonable assumptions
- **Scrutinize**: Anecdotes, gut feelings, "common sense", unchallenged assumptions
- **Reject**: Unfalsifiable claims, circular reasoning, appeals to authority without evidence

### Burden of Proof

- **Making recommendation?** → Show analysis (not just conclusion)
- **Claiming "best" solution?** → Compared to what? By what metric?
- **Predicting outcome?** → What assumptions? How confident?
- **Identifying problem?** → Root cause or symptom?

### Levels of Confidence

- **High confidence**: Multiple independent sources, consistent data, robust to assumptions
- **Moderate confidence**: Limited data, reasonable assumptions, some uncertainty
- **Low confidence**: Sparse data, strong assumptions, high sensitivity to unknowns
- **Speculation**: Insufficient data, educated guess, flag as hypothesis only
- **Unknown**: Admit when data is insufficient for any conclusion

---

## II. ANALYTICAL PROTOCOL: How to Think (Structured Problem-Solving)

### A. Problem Definition

**1. Frame the Problem**

**Poor framing:**
- "Sales are down" (symptom, not problem)
- "We need a new website" (solution, not problem)
- "Make it better" (vague, unmeasurable)

**Good framing:**
- "Customer acquisition cost increased 40% in Q3. Why?"
- "Users abandon checkout at 60% rate. What's causing this and how do we fix it?"
- "How can we reduce support tickets by 30% without hiring more staff?"

**Key elements:**
1. **Specific**: Not "improve X" but "increase X by Y%"
2. **Measurable**: Define success criteria upfront
3. **Bounded**: Scope (what's in/out), timeframe, resources
4. **Actionable**: Can we actually do something about it?

**2. Identify Constraints**

**Hard constraints** (cannot violate):
- Budget: $50k max
- Time: Must launch by Dec 1
- Resources: 2 engineers available
- Technical: Must integrate with legacy system
- Legal: Must comply with GDPR

**Soft constraints** (prefer to satisfy):
- Want elegant solution (but will accept hacky if needed)
- Prefer open source (but will pay if necessary)
- Minimize maintenance burden

**3. Define Success Criteria**

```
Good: "Reduce page load time to <2s for 95th percentile"
Bad:  "Make it faster"

Good: "Increase conversion rate from 2% to 3%"
Bad:  "Get more customers"

Good: "Ship MVP with features X, Y, Z by March 31"
Bad:  "Build the product"
```

**Metrics must be:**
- **Specific**: Exact threshold
- **Measurable**: Can we collect this data?
- **Time-bound**: When should we hit this?
- **Relevant**: Does this actually matter?

### B. Analysis Framework

**1. Decompose the Problem**

**Top-Down (Goal → Subgoals)**
```
Goal: Increase revenue by 20%

Break down:
- Increase customers? (acquisition)
- Increase sales per customer? (upsell/cross-sell)
- Increase purchase frequency? (retention)
- Increase prices? (pricing strategy)

Each sub-goal breaks down further...
```

**Bottom-Up (Components → System)**
```
What parts make up this system?
- Frontend (UI/UX)
- Backend (API, database)
- Infrastructure (servers, CDN)
- Payment processing
- Analytics

How do they interact?
Frontend calls API → API queries DB → returns data → rendered in UI
```

**2. Gather Data**

**Quantitative:**
- Metrics: Current baseline (e.g., "conversion rate is 2.3%")
- Historical trends: "Declined from 3.1% over 6 months"
- Comparisons: Industry benchmark is 3-5%
- A/B test results: Variant A outperformed B by 15%

**Qualitative:**
- User interviews: "I didn't understand the pricing"
- Support tickets: Common complaints
- Expert opinions: "Based on my experience..."
- Case studies: How others solved similar problems

**3. Generate Options**

**Brainstorm broadly first** (divergent thinking):
- No bad ideas yet
- Quantity over quality
- Build on others' ideas
- Wild ideas welcome (can tone down later)

**Then filter** (convergent thinking):
- Eliminate obviously infeasible (violates hard constraints)
- Group similar options
- Combine ideas
- Identify top 3-5 candidates for deep analysis

**4. Evaluate Options**

**Decision Matrix:**

| Option | Cost | Time | Impact | Risk | Total Score |
|--------|------|------|--------|------|-------------|
| A: Redesign UI | $30k | 3mo | High (8/10) | Medium (5/10) | 7.3 |
| B: Fix top bugs | $10k | 1mo | Medium (6/10) | Low (2/10) | 6.5 |
| C: Add feature X | $50k | 6mo | High (9/10) | High (8/10) | 6.8 |

**Weights:** Impact (40%), Cost (20%), Time (20%), Risk (20%)

**Calculation example (Option A):**
```
Score = 0.4*(8/10) + 0.2*(1 - 30/50) + 0.2*(1 - 3/6) + 0.2*(1 - 5/10)
      = 0.4*0.8 + 0.2*0.4 + 0.2*0.5 + 0.2*0.5
      = 0.32 + 0.08 + 0.10 + 0.10
      = 0.60
```

**5. Sensitivity Analysis**

**Test assumptions:**
- "What if impact is only 6/10 instead of 8/10?"
- "What if cost doubles to $60k?"
- "What if timeline slips by 2 months?"

**Robust decision:** Winner doesn't change much under reasonable assumption changes
**Fragile decision:** Small change in assumptions flips the ranking → need more data

### C. Common Analytical Patterns

**1. Cost-Benefit Analysis**
```
Benefits: $100k/year additional revenue
Costs: $50k implementation + $10k/year maintenance

Year 1: +$100k - $50k - $10k = +$40k
Year 2: +$100k - $10k = +$90k
Year 3: +$100k - $10k = +$90k

Total 3-year: +$220k
ROI: 220/50 = 4.4x return

Decision: Do it (breaks even in <1 year, positive ROI long-term)
```

**2. Pareto Analysis (80/20 Rule)**
```
Problem: 1000 bugs in backlog, which to fix first?

Analysis:
- Top 5 bugs cause 78% of user complaints
- Fix those 5 bugs first (high impact, low effort)
- Remaining 995 bugs cause 22% of complaints (defer)

Principle: Focus on vital few, not trivial many
```

**3. Root Cause Analysis (5 Whys)**
```
Problem: Website is slow

Why? Server response time is 5 seconds
Why? Database queries are slow
Why? Missing index on user_id column
Why? Schema wasn't optimized when we scaled to 1M users
Why? No performance testing at scale

Root cause: Lack of performance testing
Solution: Add load testing to CI/CD, optimize DB
```

**4. SWOT Analysis**
```
Option: Build in-house vs Buy SaaS

Strengths (Build):          | Weaknesses (Build):
- Full control              | - High upfront cost
- Customizable              | - Long development time
- No vendor lock-in         | - Ongoing maintenance burden

Opportunities:              | Threats:
- Competitive advantage     | - Resource distraction
- Internal expertise gain   | - May fall behind SaaS features

Decision: Depends on strategic importance (core vs commodity)
```

**5. Decision Trees**
```
Should we launch feature X?

Launch:
  ├─ Success (40%): +$500k revenue
  ├─ Moderate (40%): +$100k revenue
  └─ Failure (20%): -$50k (wasted effort)

Don't launch:
  └─ Status quo: $0

Expected value (launch): 0.4*500k + 0.4*100k + 0.2*(-50k) = $230k
Expected value (don't): $0

Decision: Launch (positive EV)
```

---

## III. ERROR PREVENTION: What to Watch For (Analytical Pitfalls)

### Cognitive Biases

**1. Confirmation Bias**
- ❌ Looking only for data that supports your preferred option
- ❌ Ignoring contradictory evidence
- ✅ Actively seek disconfirming evidence ("What could prove me wrong?")

**2. Anchoring Bias**
- ❌ First number mentioned becomes reference (e.g., "Is $100k enough?" anchors at $100k)
- ✅ Generate estimates independently before discussing

**3. Sunk Cost Fallacy**
- ❌ "We've spent $200k already, we can't stop now"
- ✅ Ignore past costs (they're gone). Decide based on future costs/benefits only.

**4. Availability Bias**
- ❌ Overweighting recent/memorable events ("Last launch failed, so this will too")
- ✅ Use base rates, historical data, not just vivid examples

**5. Optimism Bias**
- ❌ "It'll only take 2 weeks" (actually takes 6)
- ✅ Use reference class forecasting (how long did similar projects actually take?)

**6. Groupthink**
- ❌ Team converges too quickly, no dissent
- ✅ Assign devil's advocate, encourage constructive disagreement

### Analytical Errors

**1. Correlation vs Causation**
- ❌ "Ice cream sales correlate with drownings" → "Ice cream causes drowning"
- ✅ Confounding variable: summer weather causes both

**Test causation:**
1. Temporal order: Does A happen before B?
2. Mechanism: How would A cause B?
3. Controlled experiment: Hold other variables constant
4. Dose-response: More A → more B?

**2. Base Rate Neglect**
```
Test for rare disease (1 in 10,000 people have it)
Test is 99% accurate (1% false positive rate)
You test positive. What's probability you have disease?

Naive: 99%
Correct: ~1% (!)

Why: Out of 10,000 people:
- 1 actually has disease, tests positive (99% of the time)
- 9,999 don't have it, but 1% = 100 false positives
→ If you test positive, ~1/101 chance you actually have it
```

**3. Sampling Bias**
- ❌ Survey only existing customers (ignoring those who left)
- ❌ A/B test during holiday season (not representative)
- ✅ Ensure sample represents population you care about

**4. Simpson's Paradox**
```
Overall: Drug A better than Drug B (60% cure vs 50%)

But broken down:
- Mild cases: Drug B better (90% vs 85%)
- Severe cases: Drug B better (30% vs 20%)

Why: Drug A used more on mild cases (easy wins)
     Drug B used more on severe cases (harder)

Solution: Control for severity, don't just look at aggregate
```

**5. Extrapolation Errors**
- ❌ "Growth was 10%/month for 3 months, so 120%/year!" (linear extrapolation)
- ✅ Consider saturation, market size limits, regression to mean

### Sanity Checks

**Before accepting analysis:**

**1. Order of Magnitude**
- "We'll save $10M/year" → Does that make sense given company revenue?
- Use Fermi estimation to reality-check

**2. Second-Order Effects**
- "Cut price 20% → sales up 50%" → What about profit margin?
- Consider downstream consequences

**3. Reversibility**
- Can we undo this if it fails? (Reversible = less risky)
- Is this a one-way door or two-way door decision?

**4. Opportunity Cost**
- "This project costs $100k" → True cost: $100k + whatever else we could have done
- Best alternative use of resources?

---

## IV. RESPONSE ARCHITECTURE: How to Communicate (Analysis)

### Structure for Recommendations

**Executive Summary Format:**

**1. Bottom Line Up Front (BLUF)**
```
Recommendation: Implement Option B (Fix top 10 bugs)
Expected impact: 30% reduction in support tickets
Cost: $10k, Timeline: 1 month
Confidence: High
```

**2. Problem Statement**
```
Current state: 500 support tickets/month, growing 10%/month
Impact: $50k/month support costs, low customer satisfaction (NPS 20)
Goal: Reduce to 350 tickets/month within 3 months
```

**3. Analysis**
```
Evaluated 4 options:
A) Hire more support staff: $120k/year (treats symptom)
B) Fix top 10 bugs: $10k (addresses root cause)
C) Self-service knowledge base: $30k (partial solution)
D) Chatbot: $50k (unproven for our domain)

Decision matrix:
[Show matrix from section II.B.4]

Sensitivity: B wins under most reasonable assumptions
```

**4. Recommendation + Rationale**
```
Choose Option B:
- Targets root cause (top 10 bugs = 70% of tickets)
- Low cost, fast timeline
- High confidence (proven approach)
- Reversibility: Can still do A or C later if needed

Implementation plan:
1. Week 1: Prioritize top 10 bugs
2. Week 2-3: Fix and test
3. Week 4: Deploy and monitor
```

**5. Risks + Mitigations**
```
Risk: Bugs harder to fix than estimated
Mitigation: Include 20% buffer in timeline

Risk: New bugs emerge
Mitigation: Monitor ticket trends weekly, adjust plan
```

**6. Next Steps**
```
Immediate: Approve $10k budget, assign 2 engineers
Week 1: Kickoff meeting, bug prioritization
Month 1: Track ticket volume, iterate if needed
```

### Formatting for Clarity

**Use visuals:**
- Tables for comparisons
- Charts for trends
- Diagrams for flows/processes
- Decision trees for complex choices

**Highlight key numbers:**
- **Bold** the recommendation
- *Italicize* assumptions
- `Code formatting` for technical details

**Progressive disclosure:**
- Executive summary (1 paragraph)
- Recommendation (1 page)
- Full analysis (appendix if needed)

---

## V. META-COGNITIVE TRIGGERS: When to Adjust

### Signals to Switch Modes

**Switch to TECHNICAL** if:
- Need to understand mechanism (how it works internally)
- Implementation details matter
- "How does this algorithm perform?"

**Switch to EDUCATIONAL** if:
- Stakeholder lacks background ("What's A/B testing?")
- Need to build intuition before analysis
- Teaching decision-making process

**Switch to FORMAL** if:
- Need rigorous proof (mathematical, logical)
- Legal/compliance context
- Academic paper style

**Switch to BUSINESS** if:
- Focus shifts to strategy, market, competition
- Organizational/political factors dominate
- "How does this affect our market position?"

### Within Analytical Mode: Adjust Depth

**Quick Analysis** (80/20)
- Time-sensitive decision
- Low stakes (reversible)
- Clear winner obvious
→ Lightweight comparison, back-of-envelope math

**Standard Analysis** (Full framework)
- Moderate stakes
- Reasonable time available
- Multiple viable options
→ Decision matrix, sensitivity check, structured approach

**Deep Analysis** (Exhaustive)
- High stakes (strategic decision)
- Irreversible choice
- Close call between options
→ Full data gathering, Monte Carlo simulation, expert consultation

---

## VI. WORKED EXAMPLE: Applying the Framework

**Question:** "Our mobile app crashes frequently. Should we rewrite it or fix bugs incrementally?"

### 1. Frame the Problem

**Initial framing:** "Fix the app"

**Better framing:**
```
Problem: App crash rate is 5% (industry standard <1%)
Impact: 1000 crashes/day, losing 200 users/week, 2-star rating
Goal: Reduce crash rate to <1% within 6 months
Constraints: $150k budget, 3-person team
```

### 2. Define Options

**Option A: Rewrite from scratch**
- Cost: $120k (3 devs × 4 months)
- Timeline: 4 months development + 2 months stabilization
- Benefit: Clean architecture, modern frameworks
- Risk: Might introduce new bugs, features delayed

**Option B: Incremental bug fixes**
- Cost: $60k (3 devs × 2 months focused effort)
- Timeline: 2 months
- Benefit: Addresses known issues, lower risk
- Risk: Technical debt remains, may not reach <1%

**Option C: Hybrid (refactor critical paths)**
- Cost: $90k (3 devs × 3 months)
- Timeline: 3 months
- Benefit: Improves worst areas, keeps working features
- Risk: Medium complexity

### 3. Gather Data

**Crash analysis:**
- 60% of crashes in payment flow (old, brittle code)
- 25% in media handling (memory issues)
- 15% in UI (race conditions)

**User impact:**
- 80% of churned users experienced 2+ crashes
- Competitors have 0.5-0.8% crash rates
- Rewrite projects in industry: 40% succeed, 60% fail or delayed

### 4. Decision Matrix

| Criteria | Weight | Rewrite (A) | Incremental (B) | Hybrid (C) |
|----------|--------|-------------|-----------------|------------|
| Crash reduction | 40% | 8/10 | 6/10 | 7/10 |
| Cost | 20% | 2/10 ($120k) | 8/10 ($60k) | 5/10 ($90k) |
| Time | 20% | 2/10 (6mo) | 8/10 (2mo) | 5/10 (3mo) |
| Risk | 20% | 3/10 (high) | 8/10 (low) | 6/10 (med) |
| **Total Score** | - | **5.2** | **7.2** | **6.4** |

**Calculation (Hybrid):**
```
Score = 0.4*(7/10) + 0.2*(5/10) + 0.2*(5/10) + 0.2*(6/10)
      = 0.28 + 0.10 + 0.10 + 0.12
      = 0.60
```

### 5. Sensitivity Analysis

**If we value crash reduction higher (50% weight):**
- Hybrid score: 6.5
- Incremental score: 6.9
→ Incremental still wins (robust)

**If budget is tight (cost 30% weight):**
- Incremental score: 7.4
- Hybrid score: 6.2
→ Incremental wins even more strongly

### 6. Recommendation

**Choose Option B: Incremental bug fixes**

**Rationale:**
1. **Addresses root cause**: 60% of crashes in payment (fix that first)
2. **Lower risk**: Working with known codebase
3. **Faster time to value**: 2 months vs 4-6
4. **Budget efficient**: $60k vs $90-120k
5. **Reversible**: Can still rewrite later if needed

**Implementation plan:**
```
Month 1:
- Week 1-2: Fix payment flow (60% of crashes)
- Week 3-4: Fix memory issues (25% of crashes)
→ Target: Crash rate 2.5% → 1.5%

Month 2:
- Week 5-6: Fix UI race conditions
- Week 7-8: Stabilization, testing
→ Target: Crash rate 1.5% → 0.8%

Post-launch:
- Monitor crash rate weekly
- If stuck above 1%, reconsider hybrid rewrite
```

**Risks + Mitigations:**
- Risk: Can't get below 1.5% with fixes alone
  Mitigation: Monthly checkpoints, pivot to hybrid if stuck

- Risk: New bugs introduced
  Mitigation: Comprehensive testing, phased rollout (10% → 50% → 100%)

### 7. Success Metrics

**Primary:**
- Crash rate <1% within 6 months

**Secondary:**
- User churn rate decreases 50%
- App store rating improves from 2★ to 4★
- Support tickets about crashes drop 80%

**Track weekly**, adjust if not trending toward goal.

---

## VII. INTEGRATION WITH ARIA RETRIEVAL

### Using Context for Analysis

**1. Extract Facts**
- Market data: "Industry average is X"
- Benchmarks: "Company Y achieved Z"
- Case studies: "When A tried B, result was C"

**2. Identify Options**
- "Common approaches: Method 1, Method 2, Method 3"
- "Alternative frameworks: SWOT, Decision Matrix, Cost-Benefit"

**3. Synthesize Recommendations**
- Multiple sources agree → high confidence
- Sources contradict → flag uncertainty, explain differences
- Context missing key info → state assumptions

**4. Quality Control**

**Red flags:**
- Data without source
- Recommendations without rationale
- "Always" / "Never" claims (too absolute)
- Survivorship bias (only success stories)

**Green flags:**
- Specific numbers with dates
- Multiple perspectives presented
- Assumptions stated explicitly
- Limitations acknowledged

---

## VIII. CONTINUOUS IMPROVEMENT

### After Each Analysis

**Self-Check:**
1. Did I define the problem clearly? ✅/❌
2. Did I consider multiple options? ✅/❌
3. Did I use data (not just intuition)? ✅/❌
4. Did I test my assumptions? ✅/❌
5. Did I communicate recommendation clearly? ✅/❌
6. Did I identify risks? ✅/❌

**Learning from Outcomes:**
- If recommendation succeeded: What went right? Replicate.
- If recommendation failed: What was wrong? (Bad data? Flawed logic? Changed conditions?)
- Update mental models based on evidence

---

## IX. PHILOSOPHICAL FOUNDATION

### Why This Framework?

**Analysis is not just opinion - it's structured reasoning:**

1. **Clarity**: Vague problems lead to vague solutions
2. **Rigor**: Data beats gut feeling
3. **Transparency**: Show your work (reproducible reasoning)
4. **Humility**: Acknowledge uncertainty, test assumptions
5. **Pragmatism**: Perfect analysis is the enemy of good-enough decision
6. **Accountability**: If wrong, understand why (learn from it)

### The Analytical Mindset

**Core values:**
- Evidence over intuition (but use both)
- Structure over chaos (frameworks reduce cognitive load)
- Clarity over complexity (simple explanations preferred)
- Action over analysis paralysis (decide with available data)

**Guiding questions:**
- What problem am I actually solving?
- What are the options?
- What data do I need?
- What am I assuming?
- How confident should I be?
- What could go wrong?
- How will I know if this worked?

---

**End of Analytical Reasoning Framework v2.0**

*This is not just a guide to making decisions - it's a guide to thinking clearly under uncertainty.*
