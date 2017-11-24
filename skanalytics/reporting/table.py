# -*- coding: utf-8 -*-
from .exception import ReportingException


class TableException(ReportingException):
    pass


class Table(object):
    def __init__(self, columns):
        if not isinstance(columns, 'list'):
            raise TableException('Table must be initialized with a list')
        self.columns = columns
