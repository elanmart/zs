class Config:

    class Source:
        wiki = 'enwiki'    # {'enwiki', 'simplewiki'}
        date = '20180520'

    class Extraction:
        lowercase   = False
        tokenize    = False
        lemmatize   = False
        replace_num = False

    class Documents:
        fields       = ['title', 'summary']  # {'title', 'summary', 'rest'} What fields to keep.
        merge_fields = False  # if True: concatenate the fields into one field named e.g. "title_summary"
        merge_token  = None   # default: <end-fieldname> inserted between merged fields' texts

        class Length:
            low         = 5
            high        = 256
            used_fields = ['summary', 'rest']  # fields to use when determining document length.
            conditions  = 'all'  # {all, concat}. If all, every used_field has to have min_length.

        class PostProcessing:
            """ Before writing the documents to final destination, apply this post processing to fields `fields` """
            lower  = None
            token  = None
            lemma  = True
            fields = ['summary']

    class Definitions:
        source = 'dictionary'  # {dictionary, wikipedia}

        class Length:
            low  = 5
            high = 256

        class Policy:
            split = False  # if true, split longer definitions into multiple items.

        class Dates:
            mode = 'ignore'  # {ignore, drop-example, drop-token}

        class PostProcessing:
            """ Before writing the definitions to final destination, apply this post processing to fields `fields` """
            lower  = None
            token  = None
            lemma  = True
            fields = ['definition']

        class DictConfig:
            """ config applied only if source == 'dictionary' """
            mode = 'first'  # {all, first, concat}

        class WikiConfig:
            """ config applied only if source == 'wikipedia'
            Will use titles of wiki articles as keys when looking up a definition for category naem.
            """
            fields      = {'summary'}  # {'summary', 'rest'}
            tokenize    = True
            lemmatize   = False
            title_lower = True         # "Titles" -> "titles"
            title_lemma = False        # "Titles" -> "title"
            remove_parenthesis = True  # "Title (disambiguation)" -> "Title"

    class Split:
        """ # train - validation - test split proportions """
        proportions = (0.75, 0.15, 0.1)

    class UnseenLabels:
        """ How to generate a set of "unseen" labels for the ZSL scenario """
        mode = 'frequency'  # {frequency, fraction}

        class FreqnecyConfig:
            """ Applied if mode == 'frequency' """
            min_freq = 2  # labels occuring less than `min_freq` times are moved to test set.

        class FranctionConfig:
            """ TODO(elanmart): this is not implemented at this moment. """
            least_frequent = True  # if True, subsample the least frequent labels
            fraction       = 0.1   # fraction of labels to choose as unseen

    class Outputs:
        """ What file formats should be generated? """
        json      = True  # Json is always generated anyway. Listed for completeness
        tensor    = True
        tfidf     = True
        coo       = True
        fasttext  = True
        starspace = True
