#!/bin/bash
# Installation script for MacOS

# Install required lanagues  using homebrew
brew install Caskroom/cask/julia
brew install python
brew install python3

# Install required Julia packages
julia -e 'Pkg.add("MNIST")'
julia -e 'Pkg.add("PyPlot")'
julia -e 'Pkg.add("PyCall")'
julia -e 'Pkg.add("IProfile")'
julia -e 'Pkg.add("Calculus")'

# Install Pip
pip install -U pip setuptools

# Install required python packages
pip install theano
pip install lasagne
