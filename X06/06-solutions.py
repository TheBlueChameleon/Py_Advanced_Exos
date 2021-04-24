import matplotlib.pyplot as plt

import numpy  as np
import pandas as pd

# ============================================================================ #
# constants, behaviour parameters

pregnancyTimeFactor = 280 / 365                                                 # the average pregnancy takes 280 days, i.e. "this many years"

# we will be using these categories in our analysis.
relevantCategories = ["Area", "Population", "Birth rate", "Infant mortality rate", "Education expenditures", "Unemployment rate"]

dataPath   = "./data/"                                                          # the actual data files are here ...
dataPrefix = "c"                                                                # begin with this, followed by a category ID, and ...
dataSuffix = ".csv"                                                             # end with this

# ============================================================================ #
# just to make the on-screen output nicer

def printSeparator (name) :
    print("# " + "=" * 76 + " #")
    print(name)
    print()
    
# ============================================================================ #
# find out which CSV files we need to load

categoryID_df = pd.read_csv("./categories.csv")
categoryIDs   = {
    category : (categoryID_df[categoryID_df["Name"] == category]["Num"]).values[0]
    for category in relevantCategories
}

filenames = {category : dataPath + dataPrefix + str(cID) + dataSuffix for category, cID in categoryIDs.items() }

print("About to load these files:")
for f in filenames :
    print(" ", f)

# ============================================================================ #
# load the relevant CSVs into a helper dict of DataFrames

sources = dict()
for cat, f in filenames.items() :
    sources[cat] = pd.read_csv(f)

print("...done.")
print()

# ============================================================================ #
# reduce the DataFrames into Series with country names as labels and convert back into a unified DataFrame

data = pd.DataFrame({
    key : x.set_index("Name")["Value"]
    for key, x in sources.items()
})

# ============================================================================ #
# drop lines with unavailable data

data = data.dropna()

# ============================================================================ #
# expand the DataFrame

data["Pregnancy density"           ] = data["Birth rate"] * data["Population"] / 1000 * pregnancyTimeFactor / data["Area"]
data["Population density"          ] = data["Population"] / data["Area"]
data["Normalized pregnancy density"] = data["Pregnancy density"] / data["Population density"]

printSeparator("All data, alphabetical order")
print(data)
print()


# ============================================================================ #
# sort values by pregnancy density

printSeparator("All data, sorted by pregnancy density")
data.sort_values("Pregnancy density", inplace=True)
print(data)
print()

print("Lowest  pregnancy density:", data["Pregnancy density"].min())
print("Highest pregnancy density:", data["Pregnancy density"].max())
print()

# ============================================================================ #
# create and show bins

printSeparator("Binning")

bins = np.geomspace(1E-4, 150, 30)
binnedData = pd.cut(data["Pregnancy density"], bins)

uniqueBins = set(binnedData)
binSizeDF = pd.DataFrame(
    [ len(data[binnedData == b]) for b in uniqueBins ],
    index = uniqueBins,
    columns = ["Size of bin"]
)
binSizeDF.sort_index(inplace=True)

print(binSizeDF)
print("Total:", binSizeDF.sum()[0])
print()

print("Pregnancy density class for Germany:", binnedData["Germany"])
print("Countries and pregnancy densities in the same class:")
filtered = data[binnedData == binnedData["Germany"]]
print(filtered["Pregnancy density"])
print()


# ============================================================================ #
# data analysis

printSeparator("Preparing Plots...")

view = data.set_index("Population density")
view["Pregnancy density"].plot(linestyle="", marker=".")

plt.xscale("log")
plt.yscale("log")
plt.ylabel("Pregnancy density")
plt.show()

view = data.set_index("Normalized pregnancy density")
view["Infant mortality rate" ].plot(linestyle="", marker=".", label = "Infant mortality rate")
view["Education expenditures"].plot(linestyle="", marker=".", label = "Education expenditures")
view["Unemployment rate"     ].plot(linestyle="", marker=".", label = "Unemployment rate")

# "Unemployment rate"]

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

print("... done")

# ============================================================================ #
# in case you feel like using a spreadsheet program to look at the results

#data.to_csv("./data.csv")
