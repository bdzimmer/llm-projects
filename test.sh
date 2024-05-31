#!/usr/bin/env bash

pytest -s --tb=short --cov-report html:htmlcov/tests --cov=projects projects

xdg-open htmlcov/tests/index.html &
open htmlcov/tests/index.html
