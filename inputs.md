# Baseline (Non Agentic) & Agentic inputs
This section covers the exact outputs fed into the models during the baseline
procedure section. It covers zero-shot and two-shot respectively, and exactly
what data is fed into the model as well.

## zero-shot & two-shot
Covers the zero-shot inputs of the model, as well as data.
The data is chosen randomly. The two-shot variant is provided for each one.

-------------------------------------------------------------------------------
**For the input, provide the model with this prompt:**

Analyze the story and finish the last sentence. Your response should only be 
ONE sentence which is the final sentence that completes the story

**and then attach the data as follows**

### ROCStories
1.  Sherry hates basketball. 
    Sherry's boyfriend Tom loves basketball. 
    Sherry tries to learn more about basketball to make Tom happy. 
    For Tom's birthday she surprises him with tickets to a game. 
    
**Ground Truth:** Sherry attends her first basketball game with Tom.

2.  Brad needed to do laundry but he felt guilty not doing a full load. 
    He saw that his hamper was only half full. 
    Brad walked through his house picking up clothes he left around. 
    He even took the tea towels from the kitchen. 
    
**Ground Truth:** Once brad was finished he had just enough for one full load.

3.  I had a date to play chess online with my friend Mick. 
    First he was an hour late because his wife was working. 
    Then we started playing. 
    Then he said he had to help his son do a project on Bosnia. 
    
**Ground Truth:** We never got to play.

4.  Rachel went to her bed friend's house for Thanksgiving. 
    Her friend's mom and dad cooked a turkey and five side dishes. 
    They all ate dinner. 
    They ate pumpkin pie for dessert. 
    
**Ground Truth:** They then socialized and laughed until Rachel went home.

-------------------------------------------------------------------------------
**For the input, provide the model with this prompt:**

Analyze the function signature and docstring, and provide a finished function. 
Your response should ONLY be the full completed function (including the signature and docstring).

**and then attach the data as follows**

### HumanEval
1. def encode(message):
    """
    Write a function that takes a message, and encodes in such a 
    way that it swaps case of all letters, replaces all vowels in 
    the message with the letter that appears 2 places ahead of that 
    vowel in the english alphabet. 
    Assume only letters. 
    
    Examples:
    >>> encode('test')
    'TGST'
    >>> encode('This is a message')
    'tHKS KS C MGSSCGG'
    """

**Ground Truth:** 
    vowels = "aeiouAEIOU"
    vowels_replace = dict([(i, chr(ord(i) + 2)) for i in vowels])
    message = message.swapcase()
    return ''.join([vowels_replace[i] if i in vowels else i for i in message])

**Test Function:**
    def check(candidate):

    # Check some simple cases
    assert candidate('TEST') == 'tgst', "This prints if this assert fails 1 (good for debugging!)"
    assert candidate('Mudasir') == 'mWDCSKR', "This prints if this assert fails 2 (good for debugging!)"
    assert candidate('YES') == 'ygs', "This prints if this assert fails 3 (good for debugging!)"
    
    # Check some edge cases that are easy to work out by hand.
    assert candidate('This is a message') == 'tHKS KS C MGSSCGG', "This prints if this assert fails 2 (also good for debugging!)"
    assert candidate("I DoNt KnOw WhAt tO WrItE") == 'k dQnT kNqW wHcT Tq wRkTg', "This prints if this assert fails 2 (also good for debugging!)"

2. def by_length(arr):
    """
    Given an array of integers, sort the integers that are between 1 and 9 inclusive,
    reverse the resulting array, and then replace each digit by its corresponding name from
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine".

    For example:
      arr = [2, 1, 1, 4, 5, 8, 2, 3]   
            -> sort arr -> [1, 1, 2, 2, 3, 4, 5, 8] 
            -> reverse arr -> [8, 5, 4, 3, 2, 2, 1, 1]
      return ["Eight", "Five", "Four", "Three", "Two", "Two", "One", "One"]
    
      If the array is empty, return an empty array:
      arr = []
      return []
    
      If the array has any strange number ignore it:
      arr = [1, -1 , 55] 
            -> sort arr -> [-1, 1, 55]
            -> reverse arr -> [55, 1, -1]
      return = ['One']
    """

**Ground Truth:**
    dic = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
    }
    sorted_arr = sorted(arr, reverse=True)
    new_arr = []
    for var in sorted_arr:
        try:
            new_arr.append(dic[var])
        except:
            pass
    return new_arr

**Test Case:**
    def check(candidate):

    # Check some simple cases
    assert True, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate([2, 1, 1, 4, 5, 8, 2, 3]) == ["Eight", "Five", "Four", "Three", "Two", "Two", "One", "One"], "Error"
    assert candidate([]) == [], "Error"
    assert candidate([1, -1 , 55]) == ['One'], "Error"

    # Check some edge cases that are easy to work out by hand.
    assert True, "This prints if this assert fails 2 (also good for debugging!)"
    assert candidate([1, -1, 3, 2]) == ["Three", "Two", "One"]
    assert candidate([9, 4, 8]) == ["Nine", "Eight", "Four"]

3. def is_simple_power(x, n):
    """Your task is to write a function that returns true if a number x is a simple
    power of n and false in other cases.
    x is a simple power of n if n**int=x
    For example:
    is_simple_power(1, 4) => true
    is_simple_power(2, 2) => true
    is_simple_power(8, 2) => true
    is_simple_power(3, 2) => false
    is_simple_power(3, 1) => false
    is_simple_power(5, 3) => false
    """

**Ground Truth:**
    if (n == 1): 
        return (x == 1) 
    power = 1
    while (power < x): 
        power = power * n 
    return (power == x) 

**Test Case:**
    
        return (x == 1) 
    power = 1
    while (power < x): 
        power = power * n 
    return (power == x) 
	
def check(candidate):

    # Check some simple cases
    assert candidate(16, 2)== True, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(143214, 16)== False, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(4, 2)==True, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(9, 3)==True, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(16, 4)==True, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(24, 2)==False, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(128, 4)==False, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(12, 6)==False, "This prints if this assert fails 1 (good for debugging!)"

    # Check some edge cases that are easy to work out by hand.
    assert candidate(1, 1)==True, "This prints if this assert fails 2 (also good for debugging!)"
    assert candidate(1, 12)==True, "This prints if this assert fails 2 (also good for debugging!)"

4. 
def largest_smallest_integers(lst):
    '''
    Create a function that returns a tuple (a, b), where 'a' is
    the largest of negative integers, and 'b' is the smallest
    of positive integers in a list.
    If there is no negative or positive integers, return them as None.

    Examples:
    largest_smallest_integers([2, 4, 1, 3, 5, 7]) == (None, 1)
    largest_smallest_integers([]) == (None, None)
    largest_smallest_integers([0]) == (None, None)
    '''

**Ground Truth:**
    smallest = list(filter(lambda x: x < 0, lst))
    largest = list(filter(lambda x: x > 0, lst))
    return (max(smallest) if smallest else None, min(largest) if largest else None)

**Test Case:**
    def check(candidate):

    # Check some simple cases
    assert candidate([2, 4, 1, 3, 5, 7]) == (None, 1)
    assert candidate([2, 4, 1, 3, 5, 7, 0]) == (None, 1)
    assert candidate([1, 3, 2, 4, 5, 6, -2]) == (-2, 1)
    assert candidate([4, 5, 3, 6, 2, 7, -7]) == (-7, 2)
    assert candidate([7, 3, 8, 4, 9, 2, 5, -9]) == (-9, 2)
    assert candidate([]) == (None, None)
    assert candidate([0]) == (None, None)
    assert candidate([-1, -3, -5, -6]) == (-1, None)
    assert candidate([-1, -3, -5, -6, 0]) == (-1, None)
    assert candidate([-6, -4, -4, -3, 1]) == (-3, 1)
    assert candidate([-6, -4, -4, -3, -100, 1]) == (-3, 1)

    # Check some edge cases that are easy to work out by hand.
    assert True

-------------------------------------------------------------------------------
**For the input, provide the model with this prompt:**

Analyze the given date and give a prediction on the Federal Funds Effective Rate for that date.
Your answer should ONLY be the floating point number prediction

**and then attach the data as follows**

**For two-shot:**
Attach these two examples for each prompt, depending on the input data:
- For 2018-11-17, attach:
    2018-11-15,2.20
    2018-11-16,2.20
- For 2017-08-15, attach:
    2017-08-13,1.16
    2017-08-14,1.16
- For 2020-03-14, attach:
    2020-03-12,1.10
    2020-03-13,1.10
- For 2018-01-04, attach:
    2018-01-02,1.42
    2018-01-03,1.42

### FRED Dataset

1. 2018-11-17: 
**Ground Truth:** 2.20

2. 2017-08-15: 
**Ground Truth:** 1.16

3. 2020-03-14: 
**Ground Truth:** 1.10

4. 2018-01-04: 
**Ground Truth:** 1.42