# -*- coding: utf-8 -*-

# TODO: agenda labels set below
label_to_index = dict()
label_to_index["Online Solidarity"] = 0
label_to_index["Political Cooperation/Engagement"] = 1
label_to_index["Noncooperation/Disengagement"] = 2
label_to_index["Nonviolent Demonstration"] = 3
label_to_index["Violent Action"] = 4
label_to_index["None of These"] = 5

# index to label map
index_to_label = dict()
index_to_label[0] = "Online Solidarity"
index_to_label[1] = "Political Cooperation/Engagement"
index_to_label[2] = "Noncooperation/Disengagement"
index_to_label[3] = "Nonviolent Demonstration"
index_to_label[4] = "Violent Action"
index_to_label[5] = "None of These"

# type to label map
type_to_label = dict()
type_to_label["online"]  = "Online Solidarity"
type_to_label["Coop"]    = "Political Cooperation/Engagement"
type_to_label["nonCoop"] = "Noncooperation/Disengagement"
type_to_label["nonV"]    = "Nonviolent Demonstration"
type_to_label["Violent"] = "Violent Action"
type_to_label["other"]   = "None of These"

# all possible 1b labels sorted in a list
ordered_agenda_labels = ["Online Solidarity",
                         "Political Cooperation/Engagement",
                         "Noncooperation/Disengagement",
                         "Nonviolent Demonstration",
                         "Violent Action",
                         "None of These"
                        ]

# all possible hypothesis for the 1b agenda task
en_hypotheses = ["The author of this text encourages readers to share information relevant to a cause, promote the positions of individuals and demonstrate support for a position on an issue.",
                 "The document encourages readers to engage in the formal political process, by voting, attending public government meetings and assemblies, to support or oppose a candidate, party, law or a political position.",
                 "The author of the text wants the readers to disengage from a normal political process in order to demonstrate opposition to the status quo on an issue or to highlight the importance of a stance.",
                 "The message motivates the readers to protest peacefully in support of or opposition to a cause.",
                 "The author rallies the audience to engage personally in violent or destructive action.",
                 "The text is about something else."
                ]

fr_hypotheses = ["L'auteur de ce texte encourage les lecteurs à partager des informations pertinentes pour une cause, à promouvoir les positions des individus et à démontrer leur soutien à une position sur une question.",
                 "Le document encourage les lecteurs à s'engager dans le processus politique formel, en votant, en assistant aux réunions et assemblées publiques du gouvernement, pour soutenir ou s'opposer à un candidat, un parti, une loi ou une position politique.",
                 "L'auteur du texte souhaite que les lecteurs se désengagent d'un processus politique normal afin de manifester leur opposition au statu quo sur une question ou de souligner l'importance d'une position.",
                 "Le message motive les lecteurs à manifester pacifiquement pour soutenir ou s'opposer à une cause.",
                 "L'auteur rallie le public à s'engager personnellement dans une action violente ou destructrice.",
                 "Le texte parle d'autre chose."
                 ]

# hypothesis to label
hypothesis_to_label = dict()
hypothesis_to_label["The author of this text encourages readers to share information relevant to a cause, promote the positions of individuals and demonstrate support for a position on an issue."] = "Online Solidarity"
hypothesis_to_label["L'auteur de ce texte encourage les lecteurs à partager des informations pertinentes pour une cause, à promouvoir les positions des individus et à démontrer leur soutien à une position sur une question."] = "Online Solidarity"

hypothesis_to_label["The document encourages readers to engage in the formal political process, by voting, attending public government meetings and assemblies, to support or oppose a candidate, party, law or a political position."] = "Political Cooperation/Engagement"
hypothesis_to_label["Le document encourage les lecteurs à s'engager dans le processus politique formel, en votant, en assistant aux réunions et assemblées publiques du gouvernement, pour soutenir ou s'opposer à un candidat, un parti, une loi ou une position politique."] = "Political Cooperation/Engagement"

hypothesis_to_label["The author of the text wants the readers to disengage from a normal political process in order to demonstrate opposition to the status quo on an issue or to highlight the importance of a stance."] = "Noncooperation/Disengagement"
hypothesis_to_label["L'auteur du texte souhaite que les lecteurs se désengagent d'un processus politique normal afin de manifester leur opposition au statu quo sur une question ou de souligner l'importance d'une position."] = "Noncooperation/Disengagement"

hypothesis_to_label["The message motivates the readers to protest peacefully in support of or opposition to a cause."] = "Nonviolent Demonstration"
hypothesis_to_label["Le message motive les lecteurs à manifester pacifiquement pour soutenir ou s'opposer à une cause."] = "Nonviolent Demonstration"

hypothesis_to_label["The author rallies the audience to engage personally in violent or destructive action."] = "Violent Action"
hypothesis_to_label["L'auteur rallie le public à s'engager personnellement dans une action violente ou destructrice."] = "Violent Action"

hypothesis_to_label["The text is about something else."] = "None of These"
hypothesis_to_label["Le texte parle d'autre chose."] = "None of These"

# labels to hypotheses map
label_to_hypotheses = dict()

# The document encourages readers to share information relevant to a cause, promote or magnify
# the positions of specific individuals, use symbols or language in online profiles to
# demonstrate support for a specific position on an issue.
en_hypothesis = "The author of this text encourages readers to share information relevant to a cause, promote the positions of individuals and demonstrate support for a position on an issue."
fr_hypothesis = "L'auteur de ce texte encourage les lecteurs à partager des informations pertinentes pour une cause, à promouvoir les positions des individus et à démontrer leur soutien à une position sur une question."
label_to_hypotheses["Online Solidarity"] = (en_hypothesis, fr_hypothesis)

# The document encourages readers to engage in the formal political process, either by voting,
# attending public government meetings, assemblies, etc., to support or oppose a candidate,
# party, law, political position, or (nominally) collective action by a government,
# including military action.
# For our purposes, messages promoting cooperation/engagement have an agenda
# that is distinct from the noncooperation/disengagement agenda below.
# While both agendas may be seen as forms of political/social expression,
# they are distinct in terms of the means of expression they promote
# (or the outcomes they seek), where cooperation/engagement can
# be seen as seeking to promote deliberate participation
# (as opposed to deliberate non-participation).
en_hypothesis = "The document encourages readers to engage in the formal political process, by voting, attending public government meetings and assemblies, to support or oppose a candidate, party, law or a political position."
fr_hypothesis = "Le document encourage les lecteurs à s'engager dans le processus politique formel, en votant, en assistant aux réunions et assemblées publiques du gouvernement, pour soutenir ou s'opposer à un candidat, un parti, une loi ou une position politique."
label_to_hypotheses["Political Cooperation/Engagement"] = (en_hypothesis, fr_hypothesis)

# The document encourages readers to disengage from a normal political, economic,
# or social process in order to demonstrate opposition to the status quo on a
# specific issue or to highlight the importance of a specific stance.
# While disengagement or noncooperation is a political act and
# similar in some ways to cooperation/engagement,
# our interest here is to identify documents that specifically promote disengagement
# as a form of action in itself, as opposed to “positively engaged” forms of
# action such as voting, lobbying, meeting with politicians or candidates, etc.
en_hypothesis = "The author of the text wants the readers to disengage from a normal political process in order to demonstrate opposition to the status quo on an issue or to highlight the importance of a stance."
fr_hypothesis = "L'auteur du texte souhaite que les lecteurs se désengagent d'un processus politique normal afin de manifester leur opposition au statu quo sur une question ou de souligner l'importance d'une position."
label_to_hypotheses["Noncooperation/Disengagement"] = (en_hypothesis, fr_hypothesis)

# The document encourages readers to protest peacefully, to attend rallies, marches,
# and other forms of mass political demonstration, etc. in support of or opposition to a cause.
# The action or demonstration urged by the document must be non-violent in nature.
en_hypothesis = "The message motivates the readers to protest peacefully in support of or opposition to a cause."
fr_hypothesis = "Le message motive les lecteurs à manifester pacifiquement pour soutenir ou s'opposer à une cause."
label_to_hypotheses["Nonviolent Demonstration"] = (en_hypothesis, fr_hypothesis)

# The document encourages readers to engage personally in violent or destructive
# action (bombing, self-immolation, destruction of property, formation of militias,
# fighting in foreign countries in a mercenary capacity, etc.). A message contains
# this type of agenda when it encourages its (non-government)
# audience to engage in violence themselves, where violence in this case
# is understood to be essentially physical rather than e.g. verbal.
# Note that (as stated above) calls for government-led military
# action are evidence of a Political Cooperation agenda even though
# they may indeed result in violence.
en_hypothesis = "The author rallies the audience to engage personally in violent or destructive action."
fr_hypothesis = "L'auteur rallie le public à s'engager personnellement dans une action violente ou destructrice."
label_to_hypotheses["Violent Action"] = (en_hypothesis, fr_hypothesis)

# Label applied when none of the above Action Agendas are present in a document.
en_hypothesis = "The text is about something else."
fr_hypothesis = "Le texte parle d'autre chose."
label_to_hypotheses["None of These"] = (en_hypothesis, fr_hypothesis)

# TODO: textual entailment label map for BERT model
te_label_to_index = {'entailment': 0, 'not_entailment': 1}
te_index_to_label = {0: 'entailment', 1: 'not_entailment'}
