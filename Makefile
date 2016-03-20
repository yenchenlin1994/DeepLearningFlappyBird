BREW_PREFIX=$(shell brew --prefix)

default: run

install:
	brew tap homebrew/bundle
	brew bundle -v
	pip install -r requirements.txt

# https://github.com/Homebrew/homebrew/blob/master/share/doc/homebrew/Homebrew-and-Python.md#site-packages-and-the-pythonpath
site-packages:
	# pyenv
	mkdir -p $(HOME)/.local/lib/python2.7/site-packages
	echo 'import site; site.addsitedir("'$(BREW_PREFIX)'/lib/python2.7/site-packages")' >> $(HOME)/.local/lib/python2.7/site-packages/homebrew.pth
	# system python
	mkdir -p $(HOME)/Library/Python/2.7/lib/python/site-packages
	echo 'import site; site.addsitedir("'$(BREW_PREFIX)'/lib/python2.7/site-packages")' >> $(HOME)/Library/Python/2.7/lib/python/site-packages/homebrew.pth

run:
	python deep_q_network.py
