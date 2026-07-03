# Baseline (Non Agentic) & Agentic inputs
This section covers the exact outputs fed into the models during the baseline
procedure section. It covers zero-shot and two-shot respectively, and exactly
what data is fed into the model as well.

**Text Vectorization model:** text-embedding-3-large-model
**Code Vectorization model:** unixcoder-base

## zero-shot & two-shot
Covers the zero-shot inputs of the model, as well as data.
The data is chosen randomly. The two-shot variant is provided for each one.

-------------------------------------------------------------------------------
**For the input, provide the model with this prompt:**

Analyze the story and finish the last sentence. Your response should only be 
ONE sentence which is the final sentence that completes the story

**and then attach the data as follows**

**For two-shot:**
Attach these 2 examples for each prompt:

Example 1:
Ed got his first credit card. 
He ate out eight times the first month he had it. 
He also bought himself some new electronics. 
He was shocked when he got his first bill. 
He vowed he would be more careful in his spending.

Example 2:
Shawn was in his apartment. 
He had the window open because it was a warm day. 
Suddenly, someone stuck their hand in the window. 
It was some kid who thought he was funny. 
Shawn told him to go away.

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
Your response should ONLY be the full completed function (not including the signature and docstring).

**and then attach the data as follows**

**For two-shot:**
Attach these examples for each prompt:

Example 1:
from typing import List
def rescale_to_unit(numbers: List[float]) -> List[float]:
    """ Given list of numbers (of at least two elements), apply a linear transform to that list,
    such that the smallest number will become 0 and the largest will become 1
    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
    [0.0, 0.25, 0.5, 0.75, 1.0]
    """

Answer:
min_number = min(numbers)
max_number = max(numbers)
return [(x - min_number) / (max_number - min_number) for x in numbers]

Example 2:
def modp(n: int, p: int):
    """Return 2^n modulo p (be aware of numerics).
    >>> modp(3, 5)
    3
    >>> modp(1101, 101)
    2
    >>> modp(0, 101)
    1
    >>> modp(3, 11)
    8
    >>> modp(100, 101)
    1
    """

Answer:
ret = 1
for i in range(n):
    ret = (2 * ret) % p
return ret

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

Do not use any tools. House prices from 2006-2010
Analyze the given columns and row and predict the house price of that row.
Your answer should ONLY be the floating point number prediction

Columns:
Order,PID,MS SubClass,MS Zoning,Lot Frontage,Lot Area,Street,Alley,Lot Shape,Land Contour,Utilities,Lot Config,Land Slope,Neighborhood,Condition 1,Condition 2,Bldg Type,House Style,Overall Qual,Overall Cond,Year Built,Year Remod/Add,Roof Style,Roof Matl,Exterior 1st,Exterior 2nd,Mas Vnr Type,Mas Vnr Area,Exter Qual,Exter Cond,Foundation,Bsmt Qual,Bsmt Cond,Bsmt Exposure,BsmtFin Type 1,BsmtFin SF 1,BsmtFin Type 2,BsmtFin SF 2,Bsmt Unf SF,Total Bsmt SF,Heating,Heating QC,Central Air,Electrical,1st Flr SF,2nd Flr SF,Low Qual Fin SF,Gr Liv Area,Bsmt Full Bath,Bsmt Half Bath,Full Bath,Half Bath,Bedroom AbvGr,Kitchen AbvGr,Kitchen Qual,TotRms AbvGrd,Functional,Fireplaces,Fireplace Qu,Garage Type,Garage Yr Blt,Garage Finish,Garage Cars,Garage Area,Garage Qual,Garage Cond,Paved Drive,Wood Deck SF,Open Porch SF,Enclosed Porch,3Ssn Porch,Screen Porch,Pool Area,Pool QC,Fence,Misc Feature,Misc Val,Mo Sold,Yr Sold,Sale Type,Sale Condition,SalePrice

**and then attach the data as follows**

**For two-shot:**
Attach these two examples for each prompt:

Example 1:
1325,0902406090,050,RM,81,12150,Pave,Grvl,Reg,Lvl,AllPub,Inside,Gtl,OldTown,Norm,Norm,1Fam,1.5Fin,5,5,1954,1954,Gable,CompShg,MetalSd,MetalSd,BrkFace,335,TA,TA,BrkTil,TA,TA,No,Unf,0,Unf,0,1050,1050,GasA,Ex,N,FuseF,1050,745,0,1795,0,0,2,0,4,1,TA,7,Typ,0,NA,Attchd,1954,Unf,1,352,Fa,TA,Y,0,0,0,0,0,0,NA,NA,NA,0,11,2008,WD ,Normal,131500

Example 2:
1493,0908128050,085,RL,90,10012,Pave,NA,Reg,Lvl,AllPub,Inside,Gtl,Edwards,Norm,Norm,1Fam,SFoyer,4,5,1972,1972,Gable,CompShg,Plywood,Plywood,None,0,TA,TA,CBlock,Gd,TA,Av,BLQ,920,Rec,180,38,1138,GasA,TA,Y,SBrkr,1181,0,0,1181,1,0,2,0,3,1,TA,6,Typ,0,NA,Detchd,1974,RFn,2,588,TA,TA,Y,0,0,180,0,0,0,NA,MnPrv,NA,0,4,2008,WD ,Normal,137500

### Ames Housing Dataset

1. 1848,0533223050,160,FV,,5105,Pave,NA,IR2,Lvl,AllPub,FR2,Gtl,Somerst,Norm,Norm,TwnhsE,2Story,7,5,2004,2004,Gable,CompShg,MetalSd,MetalSd,None,0,Gd,TA,PConc,Gd,TA,No,GLQ,239,Unf,0,312,551,GasA,Ex,Y,SBrkr,551,551,0,1102,0,0,2,1,2,1,Gd,4,Typ,0,NA,Detchd,2004,Unf,2,480,TA,TA,Y,0,60,0,0,0,0,NA,NA,NA,0,3,2007,WD ,Normal
**Ground Truth:** 148800

2. 1999,0902332030,190,C (all),60,7200,Pave,NA,Reg,Lvl,AllPub,Corner,Gtl,OldTown,Norm,Norm,2fmCon,2.5Unf,6,6,1910,1998,Hip,CompShg,MetalSd,MetalSd,None,0,TA,TA,BrkTil,TA,Fa,Mn,Rec,1046,Unf,0,168,1214,GasW,Ex,N,SBrkr,1260,1031,0,2291,0,1,2,0,4,2,TA,9,Typ,1,Gd,Detchd,1900,Unf,2,506,TA,TA,Y,0,0,0,0,99,0,NA,NA,NA,0,11,2007,WD ,Normal,
**Ground Truth:** 133900

3. 1775,0528376010,060,RL,82,9044,Pave,NA,IR1,Lvl,AllPub,Inside,Gtl,NoRidge,Norm,Norm,1Fam,2Story,8,5,1996,1997,Gable,CompShg,VinylSd,VinylSd,BrkFace,526,Gd,Gd,PConc,Gd,TA,No,GLQ,1225,Unf,0,100,1325,GasA,Ex,Y,SBrkr,1335,1203,0,2538,0,0,2,1,4,1,Gd,8,Typ,1,TA,Attchd,1996,RFn,3,933,TA,TA,Y,198,92,0,0,0,0,NA,NA,NA,0,5,2007,WD ,Normal,
**Ground Truth:** 330000

4. 554,0531479020,045,RH,60,9000,Pave,NA,Reg,Lvl,AllPub,Corner,Gtl,SawyerW,Norm,Norm,1Fam,1.5Unf,6,3,1928,1950,Gable,CompShg,Wd Sdng,Wd Sdng,None,0,TA,TA,BrkTil,Fa,Fa,No,Unf,0,Unf,0,784,784,GasA,TA,N,FuseA,784,0,0,784,0,0,1,0,2,1,TA,5,Typ,0,NA,Detchd,1950,Unf,2,360,Fa,Fa,N,0,0,91,0,0,0,NA,NA,NA,0,10,2009,WD ,Normal,
**Ground Truth:** 76000