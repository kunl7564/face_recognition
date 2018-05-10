#!/usr/bin/python
# coding=utf-8
import os;

def getFileNameAndExt(filepath):
    (filepath, tempfilename) = os.path.split(filepath);
    (shotname, extension) = os.path.splitext(tempfilename);
    return shotname, extension

# print getFileNameAndExt("c:\\xx.jpg")[0]
