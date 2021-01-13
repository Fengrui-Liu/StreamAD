content = [
    ".. image:: docs/source/images/logo_htmlwithname.svg\n    :align: center",
    "docs/source/badges.rst",
    "docs/source/feature.rst",
    "docs/source/installation.rst",
    "docs/source/versioning.rst",
    "docs/source/model.rst",
]

res = ""

for item in content:
    if item.endswith(".rst"):
        f = open(item, "r").read()
        res += f

    else:
        res += item

    res += "\n"
    res += "\n"

with open("README.rst", "w+") as f:
    f.write(res)

f.close()