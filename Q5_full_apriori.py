# ==============================
# Q5: Association Rule Mining
# Complete Manual Apriori
# ==============================

from itertools import combinations

# ----------------------------------
# A. DATASET (Grocery Transactions)
# ----------------------------------

transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

print("\nTransactions:")
for t in transactions:
    print(t)

N = len(transactions)

# ----------------------------------
# B. SUPPORT FUNCTION
# ----------------------------------

def calculate_support(itemset):
    count = 0
    for transaction in transactions:
        if set(itemset).issubset(transaction):
            count += 1
    return count / N


# ----------------------------------
# C. APRIORI ALGORITHM
# ----------------------------------

def apriori(min_support):

    items = sorted(set(item for t in transactions for item in t))
    k = 1
    frequent_itemsets = {}

    while True:

        candidates = list(combinations(items, k))
        print(f"\nCandidate {k}-itemsets:", candidates)

        frequent_k = []

        for candidate in candidates:
            sup = calculate_support(candidate)

            if sup >= min_support:
                frequent_k.append(candidate)
                frequent_itemsets[candidate] = sup

        if not frequent_k:
            break

        print(f"\nFrequent {k}-itemsets:")
        for f in frequent_k:
            print(f, "Support =", round(frequent_itemsets[f], 2))

        k += 1

    return frequent_itemsets


# ----------------------------------
# D. ASSOCIATION RULE GENERATION
# ----------------------------------

def generate_rules(frequent_itemsets, min_confidence):

    print("\n==============================")
    print("Association Rules")
    print("==============================")

    for itemset in frequent_itemsets:

        if len(itemset) < 2:
            continue

        itemset_support = frequent_itemsets[itemset]

        for i in range(1, len(itemset)):
            for A in combinations(itemset, i):

                B = tuple(set(itemset) - set(A))

                support_A = frequent_itemsets.get(A, calculate_support(A))
                support_B = frequent_itemsets.get(B, calculate_support(B))

                confidence = itemset_support / support_A
                lift = confidence / support_B

                if confidence >= min_confidence:
                    print(f"{A} -> {B}")
                    print("Support =", round(itemset_support, 2))
                    print("Confidence =", round(confidence, 2))
                    print("Lift =", round(lift, 2))
                    print("------------------------------")


# ----------------------------------
# MAIN PROGRAM
# ----------------------------------

min_support = 0.4
min_confidence = 0.6

frequent_itemsets = apriori(min_support)

print("\n==============================")
print("All Frequent Itemsets")
print("==============================")
for itemset, sup in frequent_itemsets.items():
    print(itemset, "Support =", round(sup, 2))

generate_rules(frequent_itemsets, min_confidence)
