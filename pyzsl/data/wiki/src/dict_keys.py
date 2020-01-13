""" Convenience classes to keep track of what fields
are available in the processed documents.
"""


class EntryKeys:
    id         = 'id'
    url        = 'url'
    title      = 'title'
    summary    = 'summary'
    rest       = 'rest'
    categories = 'categories'


class DefinitionKeys:
    name        = 'name'
    definition  = 'definition'


class Entry:
    def __init__(self, dct):
        self.dct        = dct
        self.id         = dct[EntryKeys.id]
        self.url        = dct[EntryKeys.url]
        self.title      = dct[EntryKeys.title]
        self.summary    = dct[EntryKeys.summary]
        self.rest       = dct[EntryKeys.rest]
        self.categories = dct[EntryKeys.categories]


class Definition:
    def __init__(self, dct):
        self.dct         = dct
        self.name        = dct[DefinitionKeys.name]
        self.definitions = dct[DefinitionKeys.definition]
