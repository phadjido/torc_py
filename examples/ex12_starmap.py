"""
(C) Copyright IBM Corporation 2019
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

"""
Task parallelism with map
"""
import torcpy as torc


def work(x, y):
    return x * y


def main():
    data = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
    results = torc.starmap(work, data, chunksize=1)
    print(results)


if __name__ == '__main__':
    torc.start(main)
