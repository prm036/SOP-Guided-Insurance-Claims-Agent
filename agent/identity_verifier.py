"""
Identity Verifier — server-side PII matching.

The LLM extracts PII from natural language via function calling.
This module matches extracted fields against the policyholder database.
The LLM NEVER decides whether verification passes — this module does.

Verification rule:  ≥ 3 PII fields must match the SAME policyholder record.
Policy number alone is supplementary (semi-public) and does not count
toward the 3-field threshold.
"""

import re


class IdentityVerifier:
    """Deterministic server-side identity verification."""

    def __init__(self, data_store):
        self.data_store = data_store

    # ---------------------------------------------------------------- API

    def verify(self, collected_fields: dict) -> dict:
        """Match collected PII against all policyholders.

        Returns dict:
          matched          (bool)  — True if ≥ 3 fields match one record
          policyholder     (dict|None)
          fields_verified  (int)
          matched_fields   (list[str])
        """
        best_match = None
        best_score = 0
        best_fields: list[str] = []

        for ph in self.data_store.get_all_policyholders():
            score, matched_fields = self._score_match(collected_fields, ph)
            if score > best_score:
                best_score = score
                best_match = ph
                best_fields = matched_fields

        return {
            "matched": best_score >= 3,
            "policyholder": best_match if best_score >= 3 else None,
            "fields_verified": best_score,
            "matched_fields": best_fields,
        }

    # ----------------------------------------------------------- Scoring

    def _score_match(self, collected: dict, policyholder: dict):
        score = 0
        matched: list[str] = []

        if collected.get("name"):
            if self._match_name(collected["name"], policyholder):
                score += 1
                matched.append("name")

        if collected.get("dob"):
            if self._match_dob(collected["dob"], policyholder.get("dob", "")):
                score += 1
                matched.append("dob")

        if collected.get("phone"):
            if self._match_phone(collected["phone"], policyholder):
                score += 1
                matched.append("phone")

        if collected.get("email"):
            if self._match_email(collected["email"], policyholder):
                score += 1
                matched.append("email")

        if collected.get("ssn_last4"):
            if self._match_ssn(collected["ssn_last4"], policyholder):
                score += 1
                matched.append("ssn_last4")

        # Policy number is supplementary — does NOT count toward the
        # 3-field threshold but helps disambiguate.

        return score, matched

    # -------------------------------------------------------- Matchers

    def _match_name(self, given_name: str, policyholder: dict) -> bool:
        """Fuzzy name matching with alias support."""
        given = self._normalize_name(given_name)
        record = self._normalize_name(policyholder.get("name", ""))

        if given == record:
            return True
        if self._levenshtein(given, record) <= 2:
            return True

        for alias in policyholder.get("name_aliases", []):
            alias_norm = self._normalize_name(alias)
            if given == alias_norm or self._levenshtein(given, alias_norm) <= 2:
                return True

        return False

    @staticmethod
    def _match_dob(given_dob: str, record_dob: str) -> bool:
        return given_dob.strip() == record_dob.strip()

    def _match_phone(self, given_phone: str, policyholder: dict) -> bool:
        given = self._strip_phone(given_phone)
        record = self._strip_phone(policyholder.get("phone", ""))

        if given == record:
            return True
        for alias in policyholder.get("phone_aliases", []):
            if given == self._strip_phone(alias):
                return True
        return False

    def _match_email(self, given_email: str, policyholder: dict) -> bool:
        given = given_email.lower().strip()
        record = policyholder.get("email", "").lower().strip()

        if given == record:
            return True
        for alias in policyholder.get("email_aliases", []):
            if given == alias.lower().strip():
                return True
        return False

    def _match_ssn(self, given_ssn: str, policyholder: dict) -> bool:
        given_clean = re.sub(r"\D", "", given_ssn)[-4:]
        record_clean = policyholder.get("id_last4", "")
        return given_clean == record_clean

    # -------------------------------------------------------- Helpers

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.lower().split())

    @staticmethod
    def _strip_phone(phone: str) -> str:
        return re.sub(r"\D", "", phone)

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return IdentityVerifier._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                curr_row.append(
                    min(
                        prev_row[j + 1] + 1,   # insertion
                        curr_row[j] + 1,        # deletion
                        prev_row[j] + (c1 != c2),  # substitution
                    )
                )
            prev_row = curr_row

        return prev_row[-1]
