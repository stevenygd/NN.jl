#!/bin/bash
# Installation script for MacOS

# Install julia using homebrew
brew install Caskroom/cask/julia

# Install required Julia packages
julia -e 'Pkg.add("MNIST")'
julia -e 'Pkg.add("PyPlot")'
julia -e 'Pkg.add("PyCall")'
julia -e 'Pkg.add("IProfile")'
