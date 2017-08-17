#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated)

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        from nltk.stem.snowball import SnowballStemmer
        stemmer=SnowballStemmer("english")
        text_string=text_string.replace('\r',' ')
        text_string=text_string.replace('  ',' ')
        text_string = text_string.replace('\n', ' ')
        # words=' '.join([stemmer.stem(i.strip())for i in text_string.split()if i!=" "])
        st = ""
        # for word in word_tokenize(text_string):
        for word in text_string.split():
            st = st + " " + (stemmer.stem(word))
        # st = ""
        # for word in word_tokenize(text_string):
        # for word in text_string.split():
        #     st = st + " " + (stemmer.stem(word))
        # words=st
        # print st.strip()==words,'\n',st,'\n',words

        # to_stem = text_string.replace('\n', ' ').split()
        # words = " ".join([stemmer.stem(word) for word in to_stem])
        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        # return words
        # text_string = " ".join(text_string.split())
        # stemmer = SnowballStemmer("english")
        # split_string = text_string.split(" ")
        # new_output = ""
        # for word in split_string:
        #     stem_word = stemmer.stem(word)
        #     new_output = new_output + " " + stem_word

            # print new_output==words,words,new_output
    # return new_output
    #     ws = text_string.split()
    #     for w in ws:
    #         if w != ' ':
    #             s = stemmer.stem(w).strip()
    #             if words != "":
    #                 words = words + ' ' + s
    #             else:
    #                 words = s
    # print words==words1,"ch"
    return st
    # return words
    # return new_output
    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

