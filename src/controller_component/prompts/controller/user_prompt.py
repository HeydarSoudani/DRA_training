# ── controller prompt variant templates ──────────────────────────────────────────────

CONTROLLER_USER_TEMPLATE_NOV = """\
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

CONTROLLER_USER_TEMPLATE_NOV_COV = """\
Original question: {original_query}
Iteration: {iter_num} out of {max_iter}

Current signals:
- doc_novelty: {doc_novelty} ({num_novel_docs} out of {num_total_docs} docs)
- criteria_coverage:
{current_criteria_coverage}

Signal history:
{signal_history}

Action history:
{action_history}

Based on the above signals and history, please select the most appropriate action from ["continue", "intervene", "stop"].\
"""

CONTROLLER_USER_TEMPLATE_NOV_SIM = """\
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

CONTROLLER_USER_TEMPLATE_NOV_COV_SIM = """\
Original question: {original_query}
Iteration: {iter_num} out of {max_iter}

Current signals:
- doc_novelty: {doc_novelty} ({num_novel_docs} out of {num_total_docs} docs)
- consec_query_sim: {consec_query_sim}
- criteria_coverage:
{current_criteria_coverage}
- orig_query_sim: {orig_query_sim}

Signal history:
{signal_history}

Action history:
{action_history}

Based on the above signals and history, please select the most appropriate action from ["continue", "intervene", "stop"].\
"""

CONTROLLER_USER_TEMPLATE_SIM = """\
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

CONTROLLER_USER_TEMPLATE_COV_SIM = """\
Original question: {original_query}
Iteration: {iter_num} out of {max_iter}

Current signals:
- criteria_coverage:
{current_criteria_coverage}
- consec_query_sim: {consec_query_sim}
- orig_query_sim: {orig_query_sim}

Signal history:
{signal_history}

Action history:
{action_history}

Based on the above signals and history, please select the most appropriate action from ["continue", "intervene", "stop"].\
"""

CONTROLLER_USER_TEMPLATES = {
    "nov": CONTROLLER_USER_TEMPLATE_NOV,
    "nov_cov": CONTROLLER_USER_TEMPLATE_NOV_COV,
    "nov_sim": CONTROLLER_USER_TEMPLATE_NOV_SIM,
    "nov_cov_sim": CONTROLLER_USER_TEMPLATE_NOV_COV_SIM,
    "sim": CONTROLLER_USER_TEMPLATE_SIM,
    "cov_sim": CONTROLLER_USER_TEMPLATE_COV_SIM,
}
