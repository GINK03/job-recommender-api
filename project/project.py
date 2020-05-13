import gzip
import pickle
from flask import Flask, request, jsonify, render_template, make_response, abort
import json
import requests
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import pandas as pd
from bs4 import BeautifulSoup
from collections import namedtuple
import sys
import glob
import os
from os import environ as E

HOME = E.get("HOME")
target_dirs = [f"{HOME}/.mnt/favs{i:02d}" for i in range(20)]

@application.route('/calc_company_similarities/<username>')
def cacl_company_similarities_(username: str) -> Any:
    """
    もしusernameのログが存在していたら、それをもとに特徴量を生成、古すぎるならばどうにかするため、ログを取得しておく
    """
