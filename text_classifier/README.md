# Text Classifier
Classifying input text (wrods, phrases, sentences, or documents) using LSTM

## Notes
* The input format is shown in the example dataset.<br>
Each line is like "[label] \t [word 1] [word 2] ..."<br>

* <S>The bi-LSTM weights are tied.</S><br>
<S>I usually use different weights for bi-LSTM, but here I just followed the default implementation.</S><br>
