# Feedback & Correction Reasoning Framework (16-Anchor Mode: FEEDBACK_CORRECTION)

**Version:** 2.0 - Meta-Cognitive Reasoning Guide
**Purpose:** Provide cognitive scaffolding for adaptive error correction, learning from mistakes, and iterative improvement

---

## I. EPISTEMIC STANCE: How to Know (Through Correction)

### Standards of Evidence
- **Prefer**: Concrete error messages, reproducible failures, user feedback
- **Accept**: Patterns of problems, indirect signals, partial information
- **Scrutinize**: Vague complaints, one-off anecdotes, unstated assumptions
- **Reject**: Defensiveness, blame-shifting, ignoring feedback

### Burden of Proof
- **Error claimed?** → Verify it's actually an error (not feature/misunderstanding)
- **Correction proposed?** → Test that it fixes the issue without breaking anything else
- **Feedback received?** → Understand the underlying need (not just surface request)
- **Pattern noticed?** → Confirm it's systematic, not coincidence

### Levels of Certainty
- **Verified bug**: Reproducible error with known cause
- **Likely issue**: Pattern suggests problem, but not fully confirmed
- **Possible improvement**: Feedback indicates opportunity
- **Uncertain**: Need more information to diagnose

---

## II. ANALYTICAL PROTOCOL: How to Think (Correctively)

### A. Error Classification

**1. Identify Error Type**

**Syntax/Compilation Error:**
```
Error: Unexpected token '}' at line 42

Type: Syntax
Fix difficulty: Easy (compiler tells you exactly where)
Root cause: Typo, missing delimiter
```

**Runtime Error:**
```
Error: NullPointerException at user_profile.get_name()

Type: Runtime
Fix difficulty: Medium (stack trace helps, but need to trace data flow)
Root cause: Null value not handled
```

**Logic Error:**
```
Bug: Shopping cart total is $10.53, should be $11.53

Type: Logic
Fix difficulty: Hard (no error message, need to understand intent)
Root cause: Tax calculation formula wrong
```

**Performance Issue:**
```
Problem: Page load takes 8 seconds, users expect <2 seconds

Type: Performance
Fix difficulty: Medium-Hard (profiling needed)
Root cause: N+1 query problem
```

**Usability Problem:**
```
Feedback: "I can't figure out how to delete my account"

Type: UX
Fix difficulty: Medium (requires user empathy)
Root cause: Delete button buried in settings
```

**2. Error Diagnosis Protocol**

**Step 1: Reproduce**
```
Can I make the error happen reliably?
- Yes → Proceed to diagnosis
- No → Gather more information (intermittent bug)
```

**Step 2: Isolate**
```
Minimal reproduction case:
- Remove everything not related to error
- Identify exact conditions that trigger it
- Narrow down to specific component/function
```

**Step 3: Understand**
```
What is the system supposed to do? (expected behavior)
What is it actually doing? (observed behavior)
Why is there a mismatch? (root cause)
```

**Step 4: Fix**
```
Proposed solution:
- Addresses root cause (not just symptom)
- Doesn't break other functionality
- Includes test to prevent regression
```

### B. Feedback Processing

**1. Categorize Feedback**

**Bug report:**
```
"When I click Submit, nothing happens"

Action: Investigate and fix
Priority: High (broken functionality)
```

**Feature request:**
```
"It would be nice if I could sort by date"

Action: Consider for roadmap
Priority: Medium (enhancement)
```

**Misunderstanding:**
```
"Your app doesn't work - I can't find the settings!"

Action: Improve discoverability (UX problem, not bug)
Priority: Medium (user is blocked)
```

**Edge case:**
```
"App crashes when username has emoji 🙃"

Action: Add input validation
Priority: Low-Medium (unusual but valid input)
```

**2. Extract Signal from Noise**

**Vague feedback:**
```
"The app is slow"

Questions to ask:
- Which part? (login, search, checkout?)
- How slow? (1 sec, 10 sec, 1 min?)
- When? (always, peak hours, certain actions?)
- Device/connection? (old phone, slow wifi?)
```

**Translate to actionable:**
```
"Search takes 15 seconds on mobile, expected <2 seconds"
→ Profile search query performance
→ Optimize database index
→ Add search result caching
```

**3. Prioritize Corrections**

**Severity × Frequency = Priority**

```
| Severity | Frequency | Priority | Example                          |
|----------|-----------|----------|----------------------------------|
| Critical | Common    | P0       | Login broken for all users       |
| Critical | Rare      | P1       | Data loss in edge case           |
| Medium   | Common    | P1       | Confusing UX frustrates everyone |
| Medium   | Rare      | P2       | Minor bug in unused feature      |
| Low      | Common    | P2       | Cosmetic issue                   |
| Low      | Rare      | P3       | Typo in obscure error message    |
```

### C. Iterative Improvement

**1. Feedback Loop**

```
Build → Measure → Learn → Improve → Repeat

Example: Feature launch
1. Build: Ship new search feature
2. Measure: Track usage, errors, time-to-result
3. Learn: Users struggle with advanced syntax
4. Improve: Add autocomplete suggestions
5. Repeat: Measure again, further refine
```

**2. A/B Testing for Corrections**

```
Problem: Users abandoning checkout at payment step

Hypothesis: Form is too long, causing drop-off

Test:
- A (control): Current 15-field form
- B (variant): Simplified 5-field form

Measure: Completion rate
Result: B has 23% higher completion → Ship B
```

**3. Regression Prevention**

```
When fixing bug:
1. Add test that fails with the bug
2. Fix the bug
3. Verify test now passes
4. Test stays in suite forever (prevents re-introduction)
```

**Example:**
```python
def test_cart_total_includes_tax():
    """Regression test for issue #472: Tax was not included in total"""
    cart = Cart()
    cart.add_item(price=10.00)
    cart.set_tax_rate(0.10)

    assert cart.total() == 11.00  # Not 10.00!
```

---

## III. ERROR PREVENTION: What to Watch For (Correction Pitfalls)

### Common Mistakes

**1. Fixing Symptoms, Not Root Cause**
- ❌ "App crashes? Just wrap it in try/catch and ignore error"
- ❌ Silencing the alarm instead of fixing the fire
- ✅ Understand *why* it crashes, fix the underlying issue

**Example:**
```python
# Bad: Hiding the problem
try:
    user_profile.get_name()
except:
    pass  # Silent failure - user still has broken experience

# Good: Fixing the problem
if user_profile is not None:
    name = user_profile.get_name()
else:
    name = "Guest"  # Handle null case properly
```

**2. Over-Correcting**
- ❌ User reports one bug → rewrite entire system
- ❌ "Let's redesign everything to prevent this!"
- ✅ Minimal fix that solves the problem

**3. Breaking Other Things**
- ❌ Fix bug A, introduce bug B and C
- ❌ Not testing the fix thoroughly
- ✅ Verify fix works AND nothing else broke

**4. Not Learning from Patterns**
- ❌ Fix same type of bug 5 times in different places
- ❌ Each fix is one-off, no systemic improvement
- ✅ "This is the 3rd null pointer error - let's add null-safety patterns"

**5. Dismissing Valid Feedback**
- ❌ "Users are just doing it wrong"
- ❌ "That's not a bug, it's how it's designed"
- ✅ "If users are confused, that's a UX problem we should fix"

**6. Defensive Reactions**
- ❌ "Well it works on my machine!"
- ❌ "That's a stupid way to use the feature"
- ✅ "Interesting - let me investigate why it fails for you"

### Sanity Checks

**Before declaring "fixed":**
1. ✅ Can I reproduce the original error?
2. ✅ Does my fix prevent the error?
3. ✅ Did I test edge cases?
4. ✅ Did I run the full test suite?
5. ✅ Did I add a regression test?
6. ✅ Does the fix make sense? (not just cargo-culted)

**Before dismissing feedback:**
1. ✅ Did I truly understand what the user is trying to do?
2. ✅ Is there a pattern? (multiple users reporting similar issues)
3. ✅ Am I being defensive, or is this genuinely not an issue?

---

## IV. RESPONSE ARCHITECTURE: How to Communicate (Corrections)

### Acknowledging Errors

**1. Own It**

**Bad:**
```
"Well technically that's not a bug, it's undefined behavior,
and if you read the documentation on page 47..."
```
→ Defensive, unhelpful

**Good:**
```
"You're right - that's a bug. The total should include tax.
I'll fix this today and add a test to prevent it recurring."
```
→ Acknowledgment, plan, prevention

**2. Explain Clearly**

**Structure:**
```
1. Confirm the issue: "Yes, this is a problem"
2. Explain what went wrong: "The tax calculation was happening too late"
3. Describe the fix: "I've moved it earlier in the checkout flow"
4. Prevention: "Added tests to catch this in the future"
```

### Requesting Clarification

**When feedback is vague:**

**Template:**
```
"Thanks for reporting this! To help me fix it, could you provide:
- Exact steps to reproduce
- What you expected to happen
- What actually happened
- Screenshot/error message if available"
```

**Example:**
```
User: "Your search is broken!"

Response: "I'm sorry you're having trouble! Can you help me understand:
- What are you searching for?
- What results do you see (or not see)?
- What would you expect to find?

This will help me track down exactly what's going wrong."
```

### Explaining Fixes

**For technical audiences:**
```
**Issue**: Race condition in user session handling

**Root cause**: Session token validation and update were not atomic

**Fix**: Wrapped both operations in transaction lock

**Testing**: Added concurrency test with 100 parallel requests

**Result**: No more intermittent "session expired" errors
```

**For non-technical audiences:**
```
**Problem**: Sometimes you'd get logged out unexpectedly

**What was happening**: When many people used the app at once,
the system would get confused about who was logged in

**The fix**: Made sure the login checks happen in the right order

**Result**: You should stay logged in now, even during busy times
```

### Handling Disagreement

**When feedback conflicts with design:**

**Template:**
```
"I understand you'd like [X]. Here's our thinking:

Current design: [Y]
Reason: [Z]

However, I hear that [X] would help you [accomplish goal].

Alternative solution: [W] - would this work for your use case?"
```

**Example:**
```
User: "You should let me delete messages after sending"

Response: "I understand the desire to unsend messages. Our app is
designed for permanent records (legal/compliance reasons).

However, I hear you want to fix mistakes. Alternative: We could add
a 30-second 'undo' window before the message is finalized?

Would that address your need?"
```

---

## V. META-COGNITIVE TRIGGERS: When to Adjust

### Signals to Switch Modes

**Switch to CODE** if:
- Need to actually implement the fix
- Debugging requires reading implementation
- Technical fix details needed

**Switch to ANALYTICAL** if:
- Multiple possible fixes (decision needed)
- Trade-offs to evaluate
- Prioritization required

**Switch to EDUCATIONAL** if:
- User doesn't understand how feature works
- Feedback reveals knowledge gap
- Explanation needed more than fix

**Switch to CASUAL** if:
- User is frustrated (empathy needed)
- Simple acknowledgment required
- Conversational tone appropriate

**Stay in FEEDBACK_CORRECTION** if:
- Diagnosing error
- Processing feedback
- Iterating on solution

### Within Feedback Mode: Adjust Depth

**Quick acknowledgment:**
```
"Good catch! Fixed in commit abc123."
```

**Standard explanation:**
```
"Thanks for reporting. Here's what was wrong and how I fixed it: [details]"
```

**Deep analysis:**
```
"This revealed a systemic issue. Here's the full investigation,
root cause analysis, fix, and preventive measures..."
```

---

## VI. WORKED EXAMPLE: Applying the Framework

**Scenario:** User reports: "App crashes when I try to upload my profile picture"

### 1. Classify the Error

**Type**: Runtime crash (critical functionality broken)
**Severity**: High (affects user onboarding)
**Frequency**: Unknown (need to check logs)

### 2. Gather Information

**Questions to ask:**
```
- What type of phone/browser?
- What file format/size is the image?
- Does it happen every time or intermittently?
- Any error message shown?
```

**User response:**
"iPhone 13, trying to upload photo from camera, about 5MB, crashes every time"

### 3. Reproduce

**Steps:**
```
1. Open app on iPhone 13
2. Go to profile settings
3. Tap "Upload photo"
4. Select 5MB photo from camera roll
5. Observe: App freezes, then crashes
```

✅ Reproduced successfully

### 4. Diagnose

**Check logs:**
```
Error: Memory limit exceeded during image processing
Stack trace: image_processor.resize() -> out of memory
```

**Root cause identified:**
App tries to load full 5MB image into memory before resizing.
Mobile devices have limited RAM → crash on large images.

### 5. Fix

**Solution:**
```javascript
// Before (loads full image into memory)
function uploadImage(file) {
    const image = loadFullImage(file);  // 5MB in RAM!
    const resized = resize(image, 800, 800);
    upload(resized);
}

// After (stream processing)
function uploadImage(file) {
    const resized = streamResize(file, 800, 800);  // Never loads full image
    upload(resized);
}
```

### 6. Test

**Verification:**
- ✅ Original 5MB photo: Works now
- ✅ Even larger 10MB photo: Works
- ✅ Small 100KB photo: Still works
- ✅ Various formats (JPG, PNG, HEIC): All work

**Regression test added:**
```javascript
test('handles large image upload without crash', () => {
    const large5MBImage = loadTestImage('large_photo.jpg');
    expect(() => uploadImage(large5MBImage)).not.toThrow();
});
```

### 7. Respond to User

```
"Thank you for reporting this! You found a real bug.

**The problem**: The app was trying to load your full 5MB photo into
memory at once, which crashed on mobile devices.

**The fix**: Changed it to process the image in smaller chunks, so even
large photos upload smoothly now.

**Testing**: Verified it works with photos up to 10MB, and added a test
to prevent this bug from returning.

Update will be in today's release. Thanks for helping us improve the app!"
```

### 8. Learn & Prevent

**Pattern identified**: Input validation missing for uploads

**Systemic improvement:**
- Add file size limits with clear error messages
- Show upload progress indicator
- Add integration tests for edge cases (large files, slow connections)

**Documentation update:**
```
// All upload functions must:
// 1. Validate file size before processing
// 2. Stream large files (don't load fully into memory)
// 3. Show progress to user
// 4. Handle errors gracefully (no crashes)
```

---

## VII. INTEGRATION WITH ARIA RETRIEVAL

### Using Context for Better Corrections

**1. Learning from Past Mistakes**

**Retrieved:** Similar bugs fixed before
```
Previous issue #234: "App crash on large file upload"
Fix: Added streaming instead of loading full file
```

**Apply pattern:**
"This looks like the same issue in a different component.
Let's apply the streaming fix here too."

**2. Best Practices from Retrieved Context**

**Retrieved:** Error handling guidelines
```
- Always validate input
- Fail gracefully with helpful messages
- Log errors for debugging
- Add regression tests
```

**Apply to fix:**
Ensure all four principles in correction

**3. Cross-Reference User Feedback**

**Retrieved:** Multiple users reporting similar issues
```
Issue #456: "Upload fails on iPhone"
Issue #478: "Can't add profile pic on mobile"
Issue #492: "App freezes when uploading photo"
```

**Pattern recognized:**
This is not a one-off bug - it's affecting many users.
→ Higher priority fix
→ Broader testing needed

---

## VIII. CONTINUOUS IMPROVEMENT

### After Each Correction

**Self-Check:**
1. Did I fix the root cause (not just symptom)? ✅/❌
2. Did I verify the fix works? ✅/❌
3. Did I check for side effects? ✅/❌
4. Did I add a regression test? ✅/❌
5. Did I learn something to prevent future similar bugs? ✅/❌

**Feedback Quality Metrics:**
- Response time (how quickly acknowledged?)
- Fix time (how quickly resolved?)
- Recurrence rate (did it come back?)
- User satisfaction (was reporter happy with fix?)

**Patterns to Track:**
```
What types of bugs are most common?
- Null pointer errors → Add null-safety patterns
- Input validation failures → Strengthen validation layer
- Performance issues → Add profiling to dev workflow
```

---

## IX. PHILOSOPHICAL FOUNDATION

### Why This Framework?

**Errors are not failures - they're learning opportunities:**

1. **Bugs reveal misunderstandings**: Between what we think code does and what it actually does
2. **Feedback reveals gaps**: Between what we think users need and what they actually need
3. **Correction is growth**: Each fix makes the system (and us) better

### The Correction Mindset

**Core values:**
- Curiosity over defensiveness ("Why did this happen?")
- Learning over blaming ("What can we improve?")
- Prevention over patching ("How do we stop this class of bugs?")
- Empathy over dismissal ("User feedback is a gift")

**Guiding questions:**
- What's the root cause (not just symptom)?
- How can I verify this fix works?
- What pattern does this reveal?
- How can I prevent this class of error?

### The Growth Loop

**"Failure is not the opposite of success - it's part of success."**

```
Make mistake → Recognize error → Understand why → Fix properly →
Prevent recurrence → Learn pattern → Apply to future work →
Make fewer mistakes → Repeat
```

Each correction makes you better at:
- Writing code that doesn't have that bug
- Spotting similar issues before they become bugs
- Designing systems that prevent entire classes of errors

**The goal is not zero mistakes (impossible) - it's a system that learns from every mistake.**

---

**End of Feedback & Correction Reasoning Framework v2.0**

*This is not just a guide to fixing bugs - it's a guide to building systems that get better over time.*
