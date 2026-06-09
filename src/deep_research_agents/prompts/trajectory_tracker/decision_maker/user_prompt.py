# ── DM prompt variant templates ──────────────────────────────────────────────

DM_USER_TEMPLATE_NOV = """\
Original question: {original_query}
Iteration: {iter_num} out of {max_iter}

Current signals:
- doc_novelty: {doc_novelty} ({num_novel_docs} out of {num_total_docs} docs)

Signal history:
{signal_history}

Action history:
{action_history}

Based on the above signals and history, please select the most appropriate action from ["continue", "intervene", "stop"].\
"""

DM_USER_TEMPLATE_NOV_COV = """\
Original question: {original_query}
Iteration: {iter_num} out of {max_iter}

Current signals:
- doc_novelty: {doc_novelty} ({num_novel_docs} out of {num_total_docs} docs)
- aspect_coverage:
{current_aspect_coverage}

Signal history:
{signal_history}

Action history:
{action_history}

Based on the above signals and history, please select the most appropriate action from ["continue", "intervene", "stop"].\
"""

DM_USER_TEMPLATE_NOV_SIM = """\
Original question: {original_query}
Iteration: {iter_num} out of {max_iter}

Current signals:
- doc_novelty: {doc_novelty} ({num_novel_docs} out of {num_total_docs} docs)
- consec_query_sim: {consec_query_sim}
- orig_query_sim: {orig_query_sim}

Signal history:
{signal_history}

Action history:
{action_history}

Based on the above signals and history, please select the most appropriate action from ["continue", "intervene", "stop"].\
"""

DM_USER_TEMPLATE_NOV_COV_SIM = """\
Original question: {original_query}
Iteration: {iter_num} out of {max_iter}

Current signals:
- doc_novelty: {doc_novelty} ({num_novel_docs} out of {num_total_docs} docs)
- consec_query_sim: {consec_query_sim}
- aspect_coverage:
{current_aspect_coverage}
- orig_query_sim: {orig_query_sim}

Signal history:
{signal_history}

Action history:
{action_history}

Based on the above signals and history, please select the most appropriate action from ["continue", "intervene", "stop"].\
"""

DM_USER_TEMPLATE_SIM = """\
Original question: {original_query}
Iteration: {iter_num} out of {max_iter}

Current signals:
- consec_query_sim: {consec_query_sim}
- orig_query_sim: {orig_query_sim}

Signal history:
{signal_history}

Action history:
{action_history}

Based on the above signals and history, please select the most appropriate action from ["continue", "intervene", "stop"].\
"""

DM_USER_TEMPLATE_COV_SIM = """\
Original question: {original_query}
Iteration: {iter_num} out of {max_iter}

Current signals:
- aspect_coverage:
{current_aspect_coverage}
- consec_query_sim: {consec_query_sim}
- orig_query_sim: {orig_query_sim}

Signal history:
{signal_history}

Action history:
{action_history}

Based on the above signals and history, please select the most appropriate action from ["continue", "intervene", "stop"].\
"""

DM_USER_TEMPLATES = {
    "nov": DM_USER_TEMPLATE_NOV,
    "nov_cov": DM_USER_TEMPLATE_NOV_COV,
    "nov_sim": DM_USER_TEMPLATE_NOV_SIM,
    "nov_cov_sim": DM_USER_TEMPLATE_NOV_COV_SIM,
    "sim": DM_USER_TEMPLATE_SIM,
    "cov_sim": DM_USER_TEMPLATE_COV_SIM,
}
