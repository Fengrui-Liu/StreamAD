# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

api:
	sphinx-apidoc -f -E -M -o ./source ../streamad

doc:
	make clean
	sphinx-autogen -o generated -t source/_templates/class.rst source/index.rst
	make html


gh-pages:
	rm -rf /tmp/gh-pages
	cp -r $(BUILDDIR)/html/ /tmp/gh-pages
	git checkout gh-pages
	rm -rf * && cp -r /tmp/gh-pages/* ./ && rm -rf /tmp/gh-pages && git add . && git commit -m "Updated gh-pages" && git push -f && git checkout main