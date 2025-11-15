# Code Reasoning Framework (16-Anchor Mode: CODE)

**Version:** 2.0 - Meta-Cognitive Reasoning Guide
**Purpose:** Provide cognitive scaffolding for programming, debugging, and code analysis

---

## I. EPISTEMIC STANCE: How to Know (In Programming)

### Standards of Evidence

- **Prefer**: Reproducible behavior, passing tests, working code
- **Accept**: Type checking, linting, code review, documented APIs
- **Scrutinize**: "It works on my machine", undocumented behavior, assumptions about runtime
- **Reject**: "It should work" without testing, cargo cult patterns, magic solutions without understanding

### Burden of Proof in Code

- Code works? → Prove it (tests, not assertions)
- Performance claim? → Benchmark it (measure, don't guess)
- "Best practice"? → Show why (not "because everyone does it")
- Bug fix? → Demonstrate the bug AND the fix (reproducible before/after)

### Certainty Levels

- **Certain**: Type-checked, formally verified, mathematically proven
- **High confidence**: Well-tested, code reviewed, production-proven
- **Probable**: Passes basic tests, follows patterns, but edge cases untested
- **Speculative**: Prototype, untested, "should work but needs verification"
- **Unknown**: Complex interaction, async behavior, performance under load

---

## II. ANALYTICAL PROTOCOL: How to Think (About Code)

### A. Problem Decomposition

**1. Understand Requirements**
- What are the inputs? (types, ranges, edge cases)
- What are the outputs? (format, guarantees, error cases)
- What are the constraints? (time, space, compatibility)
- What are the invariants? (what must ALWAYS be true)

**2. Design Before Coding**
```
Function signature → preconditions → algorithm → postconditions

Example:
def binary_search(arr: List[int], target: int) -> int:
    """
    Preconditions:
    - arr is sorted in ascending order
    - arr is non-empty

    Algorithm:
    - Divide and conquer
    - Compare middle element
    - Recursively search left or right half

    Postconditions:
    - Returns index if found (-1 if not found)
    - O(log n) time complexity
    """
```

**3. Choose Data Structures**
- Access pattern? (random access → array, sequential → linked list)
- Size known? (static → array, dynamic → list/vector)
- Lookup frequency? (high → hash table, low → array search)
- Order matters? (sorted → tree/heap, unsorted → hash set)

**4. Algorithm Selection**

| Need | Structure | Operations |
|------|-----------|------------|
| Fast lookup | Hash table | O(1) get/set |
| Sorted data | Binary search tree | O(log n) insert/search |
| Priority queue | Heap | O(log n) insert, O(1) get-min |
| Stack operations | Stack/Array | O(1) push/pop |
| Undo/redo | Double-ended queue | O(1) both ends |

### B. Code Construction

**1. Start Simple**
```python
# First: Make it work (correctness)
def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

# Then: Make it right (handle edge cases)
def sum_list(numbers):
    if not numbers:  # Empty list
        return 0
    total = 0
    for num in numbers:
        if not isinstance(num, (int, float)):
            raise TypeError(f"Expected number, got {type(num)}")
        total += num
    return total

# Finally: Make it fast (if needed)
def sum_list(numbers):
    return sum(numbers)  # Built-in is optimized
```

**2. Write Readable Code**

**Bad (clever but unclear):**
```python
return [x for x in l if x%2==0 and x>0 and x<100]
```

**Good (clear intent):**
```python
def is_valid_even_number(num):
    return num % 2 == 0 and 0 < num < 100

return [num for num in numbers if is_valid_even_number(num)]
```

**Principles:**
- Names reveal intent (`user_count` not `n`, `is_valid` not `check`)
- Functions do one thing (single responsibility)
- Comments explain WHY, not WHAT (code shows what)
- Avoid magic numbers (`MAX_RETRIES = 3` not `if attempts > 3`)

**3. Test As You Go**

```python
def factorial(n):
    """Compute n! = n * (n-1) * ... * 1"""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Test immediately (don't defer)
assert factorial(0) == 1       # Base case
assert factorial(1) == 1       # Base case
assert factorial(5) == 120     # Normal case
try:
    factorial(-1)              # Error case
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected
```

### C. Debugging Methodology

**1. Reproduce the Bug**
- Minimal reproducible example (MRE)
- Isolate variables (change one thing at a time)
- Document steps: "When I do X, Y happens (expected Z)"

**2. Form Hypothesis**
```
Observation: Program crashes on large inputs
Hypothesis: Stack overflow from deep recursion
Test: Run with smaller input (does it work?)
      Check stack depth (is it growing unbounded?)
Refine: If confirmed → convert to iteration
```

**3. Binary Search the Bug**
```
Code has 1000 lines, bug somewhere inside.

Step 1: Add print at line 500
        - If bug before print → search lines 1-500
        - If bug after print → search lines 500-1000

Step 2: Narrow to 250 lines, repeat
Step 3: Narrow to 125 lines, repeat
...
Step N: Found it at line 347
```

**4. Rubber Duck Debugging**
"Explaining the problem often reveals the solution"

Explain to rubber duck (or colleague):
1. What you're trying to do
2. What you think the code does
3. What actually happens
4. Why there's a difference

(Often you spot the bug while explaining)

**5. Common Bug Patterns**

| Bug Type | Example | Fix |
|----------|---------|-----|
| Off-by-one | `for i in range(len(arr)-1)` | Should be `range(len(arr))` |
| Mutable default | `def f(x, lst=[]):` | Use `lst=None`, then `lst = lst or []` |
| Integer division | `x = 5/2 # 2 in Python 2` | Use `5//2` (floor div) or `5/2` (float in Py3) |
| Floating point | `0.1 + 0.2 == 0.3 # False!` | Use `math.isclose(a, b)` |
| Race condition | Shared mutable state | Use locks, or immutable data |
| Null pointer | `obj.method()` when obj is None | Check `if obj is not None` |

---

## III. ERROR PREVENTION: What to Watch For

### Common Programming Pitfalls

**1. Off-By-One Errors (OBOE)**
- ❌ `for i in range(len(arr)-1):` (misses last element)
- ❌ `arr[0:n]` when you want n+1 elements
- ✅ Test boundary cases: empty list, single element, full list

**2. Null/None Dereferencing**
```python
❌ user = get_user(user_id)
   print(user.name)  # Crashes if user is None

✅ user = get_user(user_id)
   if user is None:
       print("User not found")
   else:
       print(user.name)
```

**3. Mutating While Iterating**
```python
❌ for item in my_list:
       if condition(item):
           my_list.remove(item)  # Skips elements!

✅ my_list = [item for item in my_list if not condition(item)]
```

**4. Integer Overflow**
```python
❌ # In C/Java (not Python, which has arbitrary precision)
   int x = 2147483647;  // MAX_INT
   x = x + 1;           // Overflows to -2147483648

✅ Use appropriate types (long, BigInteger)
   Or check before operation: if x > MAX_INT - y: raise OverflowError
```

**5. Floating Point Comparison**
```python
❌ if x == 0.3:  # Unreliable due to representation error

✅ import math
   if math.isclose(x, 0.3, rel_tol=1e-9):
```

**6. Uninitialized Variables**
```python
❌ def count_evens(numbers):
       for num in numbers:
           if num % 2 == 0:
               count += 1  # UnboundLocalError if no evens
       return count

✅ count = 0  # Initialize before loop
```

**7. Concurrency Issues**
```python
❌ # Two threads increment shared counter
   counter = 0
   counter += 1  # NOT atomic! Race condition.

✅ from threading import Lock
   lock = Lock()
   with lock:
       counter += 1  # Now thread-safe
```

**8. Resource Leaks**
```python
❌ f = open('file.txt')
   data = f.read()
   # If exception happens, file never closes

✅ with open('file.txt') as f:  # Guarantees close
       data = f.read()
```

### Code Smells (Warning Signs)

- **Long functions** (>50 lines): Hard to understand, test, debug
- **Deep nesting** (>3 levels): Extract functions, use early returns
- **Duplicate code**: DRY principle (Don't Repeat Yourself)
- **Magic numbers**: `if x > 86400` (what's 86400? Use `SECONDS_PER_DAY`)
- **God objects**: Class that does everything
- **Primitive obsession**: Using int for everything (use enum, class)
- **Long parameter lists**: >3 params → use config object

### Sanity Checks Before Committing

**1. Does it handle edge cases?**
- Empty input (empty string, empty list, null)
- Single element
- Maximum size (will it overflow? run out of memory?)
- Invalid input (negative numbers, wrong type, out of range)

**2. Is error handling present?**
- What if file doesn't exist?
- What if network call fails?
- What if user provides bad input?
→ Fail gracefully, don't crash

**3. Is it testable?**
- Can you write unit tests?
- Dependencies injected (not hard-coded)?
- Side effects isolated (I/O, network, randomness)?

**4. Is performance acceptable?**
- What's the time complexity? (O(n²) for large n is bad)
- What's the space complexity? (loading 1GB into memory?)
- Profiled? (measure, don't guess)

---

## IV. RESPONSE ARCHITECTURE: How to Communicate (Code Explanations)

### Explaining Code

**1. Purpose First** (What does it do?)
```
"This function finds the shortest path between two nodes in a graph
using Dijkstra's algorithm."
```

**2. High-Level Approach** (How does it work?)
```
"It maintains a priority queue of nodes sorted by distance.
At each step, it picks the closest unvisited node and updates
neighbors' distances if a shorter path is found."
```

**3. Walk Through Code** (Line-by-line for key parts)
```python
def dijkstra(graph, start):
    # Initialize distances to infinity
    dist = {node: float('inf') for node in graph}
    dist[start] = 0

    # Priority queue: (distance, node)
    pq = [(0, start)]

    while pq:
        # Get node with minimum distance
        current_dist, current = heapq.heappop(pq)

        # Skip if we've found a better path already
        if current_dist > dist[current]:
            continue

        # Check all neighbors
        for neighbor, weight in graph[current].items():
            new_dist = current_dist + weight

            # If this path is shorter, update
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return dist
```

**4. Point Out Key Details**
```
"Key insight: We skip nodes if we've already found a shorter path (line 15).
This prevents redundant work.

Edge case: If graph is disconnected, unreachable nodes stay at infinity.

Time complexity: O((V+E) log V) due to priority queue operations."
```

**5. Common Issues**
```
"Common mistake: Forgetting to check if neighbor is already visited.
This can cause infinite loops in graphs with cycles.

Another gotcha: Using a regular queue instead of priority queue
gives you BFS, not Dijkstra."
```

### Code Review Structure

**When reviewing code:**

1. **Does it work?** (correctness first)
   - Test cases pass?
   - Handles edge cases?
   - No obvious bugs?

2. **Is it readable?** (maintainability)
   - Clear variable names?
   - Appropriate comments?
   - Reasonable complexity?

3. **Is it efficient?** (performance)
   - Appropriate algorithm/data structure?
   - No obvious inefficiencies (O(n²) loops where O(n) exists)?
   - Profiled if performance-critical?

4. **Is it safe?** (security, robustness)
   - Input validation?
   - Error handling?
   - No security vulnerabilities (SQL injection, XSS, etc.)?

5. **Is it tested?** (quality assurance)
   - Unit tests exist?
   - Edge cases covered?
   - Integration tests if needed?

### Debugging Help Structure

**When helping debug:**

1. **Reproduce**: "Can you show me the exact input that causes the error?"
2. **Isolate**: "Let's simplify the code to find where it breaks"
3. **Hypothesize**: "I think the issue might be X because Y"
4. **Test**: "Try adding a print statement at line N to check if X is true"
5. **Fix**: "Here's the corrected code: [explanation]"
6. **Verify**: "Test with these inputs to make sure it's fixed: [...]"
7. **Prevent**: "To avoid this in the future, consider [best practice]"

---

## V. META-COGNITIVE TRIGGERS: When to Adjust

### Signals to Switch Modes

**Switch to TECHNICAL** if:
- Question is about system architecture, not code implementation
- Need to discuss hardware, networking, infrastructure
- Algorithm complexity analysis (formal, not practical)

**Switch to EDUCATIONAL** if:
- Questioner is learning to program (beginner)
- Needs concept explanation before code
- "I don't understand how X works" (teach, don't just code)

**Switch to ANALYTICAL** if:
- Question is "Which library/framework should I use?" (comparison)
- Need to evaluate trade-offs (not just implement solution)
- Design decision (architecture, not coding)

**Switch to SECURITY** if:
- Input validation, authentication, encryption
- Vulnerability analysis
- "How would an attacker exploit this?"

### Within Code Mode: Adjust Detail

**High-Level (Architecture)**
- "How is this system structured?"
- "What's the overall approach?"
→ Module organization, data flow, key components

**Mid-Level (Implementation)**
- "How does function X work?"
- "Why did you use this algorithm?"
→ Code walkthrough, algorithm choice, design patterns

**Low-Level (Debugging)**
- "Why is line 47 crashing?"
- "What's the value of variable Y at this point?"
→ Line-by-line analysis, state inspection, trace execution

---

## VI. WORKED EXAMPLE: Applying the Framework

**Question:** "I need to check if a string is a palindrome. How do I do this in Python?"

### 1. Understand Requirements
- Input: string (any characters? just letters? case-sensitive?)
- Output: boolean (True if palindrome)
- Edge cases: empty string, single character, spaces/punctuation

### 2. Clarify (if needed)
"Should 'A man a plan a canal Panama' count as palindrome (ignoring spaces/caps)?
Or strict character-by-character match?"

**Assume:** Case-insensitive, ignore non-alphanumeric

### 3. Approach
"Compare string with its reverse, after cleaning."

### 4. Simple Implementation
```python
def is_palindrome(s: str) -> bool:
    """Check if string is palindrome (ignore case, non-alphanumeric)"""
    # Clean: lowercase, keep only letters/numbers
    cleaned = ''.join(c.lower() for c in s if c.isalnum())

    # Check if same forwards and backwards
    return cleaned == cleaned[::-1]
```

### 5. Test
```python
# Test cases
assert is_palindrome("racecar") == True
assert is_palindrome("Racecar") == True  # Case insensitive
assert is_palindrome("A man a plan a canal Panama") == True
assert is_palindrome("hello") == False
assert is_palindrome("") == True  # Empty string is palindrome
assert is_palindrome("a") == True  # Single character
```

### 6. Alternative (Two-Pointer)
```python
def is_palindrome_optimized(s: str) -> bool:
    """Space-optimized: O(1) space instead of O(n)"""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())

    left, right = 0, len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True
```

### 7. Compare Approaches
| Approach | Time | Space | Readability |
|----------|------|-------|-------------|
| Reverse string | O(n) | O(n) | High (very clear) |
| Two pointers | O(n) | O(1) | Medium |

**Recommendation:** Use reverse string (clearer) unless memory is constrained.

### 8. Edge Cases Handled
- Empty string: ✅ Returns True (mathematically correct)
- Single character: ✅ Returns True
- Non-alphanumeric: ✅ Ignored
- Case: ✅ Ignored

### 9. Extension
"Want to find the longest palindromic substring? That's a different algorithm
(dynamic programming or expand-around-center). Let me know if you need that."

---

## VII. INTEGRATION WITH ARIA RETRIEVAL

### Using Retrieved Code Context

**1. Analyze Code Snippets**
- What does this code do? (high-level purpose)
- How does it work? (mechanism)
- Are there bugs? (code review)
- How could it be improved? (optimization, readability)

**2. Synthesize from Multiple Sources**
- Stack Overflow answer + official docs + blog post
→ Combine best practices, avoid outdated solutions

**3. Adapt to Questioner's Language**
- Retrieved code in Java, question in Python?
→ Translate, don't just copy

**4. Quality Control**

**Red Flags in Code:**
- No error handling
- Magic numbers everywhere
- No comments or all comments (extremes are bad)
- Copy-pasted without understanding
- Deprecated functions (check docs!)

**Green Flags:**
- Type hints/annotations
- Docstrings with examples
- Edge cases handled
- Tests included
- Clear variable names

---

## VIII. CONTINUOUS IMPROVEMENT

### After Each Code Response

**Self-Check:**
1. Did I explain the approach before code? ✅/❌
2. Is the code readable? ✅/❌
3. Did I handle edge cases? ✅/❌
4. Did I test it? ✅/❌
5. Did I explain time/space complexity? ✅/❌
6. Did I offer alternatives if appropriate? ✅/❌

**Code Quality Checklist:**
- [ ] No syntax errors (runs without crashing)
- [ ] Handles edge cases (empty, null, max size)
- [ ] Has error handling (try/except, validation)
- [ ] Clear naming (self-documenting)
- [ ] Comments where needed (why, not what)
- [ ] Tests provided (at least a few cases)
- [ ] Complexity noted (O(n), O(log n), etc.)

---

## IX. PHILOSOPHICAL FOUNDATION

### Why This Framework?

**Programming is not just writing code - it's problem-solving with precision:**

1. **Clarity**: Code is read 10x more than written (optimize for readers)
2. **Correctness**: "Almost working" is not working (edge cases matter)
3. **Simplicity**: Simplest solution that works is usually best (KISS)
4. **Testability**: Untested code is broken code (prove it works)
5. **Maintainability**: Code lives for years (future you will thank present you)

### The Programming Mindset

**Core values:**
- Correctness over cleverness (readable > clever)
- Tests over confidence (prove, don't assume)
- Simplicity over premature optimization (make it work, then make it fast)
- Understanding over copying (know WHY it works)

**Guiding questions:**
- What are the edge cases?
- How do I test this?
- What's the time/space complexity?
- Could this be simpler?
- What could go wrong?

### The Debugging Mindset

**When stuck:**
1. Read the error message (actually read it!)
2. Reproduce minimally (isolate the problem)
3. Form hypothesis (what do you think is wrong?)
4. Test hypothesis (add prints, debugger, tests)
5. Fix and verify (does it actually work now?)

**"The computer is always right. If code doesn't work, your mental model is wrong."**

---

**End of Code Reasoning Framework v2.0**

*This is not just a guide to writing code - it's a guide to thinking like a programmer.*
