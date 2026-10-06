"""
Fixture Data Store — loads and indexes all JSON fixture data.

Provides typed query methods for policyholders, claims, document guidelines,
and representatives. All data is loaded once at startup and indexed for
fast lookups.
"""

import json
import os


class DataStore:
    """Loads and indexes fixture data for the insurance claims agent."""

    def __init__(self, fixtures_dir=None):
        if fixtures_dir is None:
            fixtures_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "fixtures"
            )

        self.fixtures_dir = fixtures_dir
        self._policyholders = []
        self._claims = []
        self._claim_schema = {}
        self._consent_scenarios = {}
        self._representatives = []
        self._document_guidelines = {}

        self._load_all()
        self._build_indexes()

    # ------------------------------------------------------------------ IO

    def _load_all(self):
        self._policyholders = self._load_json("policyholders.json")
        self._claims = self._load_json("claims.json")
        self._claim_schema = self._load_json("claim_schema.json")
        self._consent_scenarios = self._load_json("consent_scenarios.json")
        self._representatives = self._load_json("representatives.json")
        self._document_guidelines = self._load_json(
            "required_document_guideline.json"
        )

    def _load_json(self, filename):
        filepath = os.path.join(self.fixtures_dir, filename)
        with open(filepath, "r") as fh:
            return json.load(fh)

    # -------------------------------------------------------------- Index

    def _build_indexes(self):
        self._ph_by_id = {ph["party_id"]: ph for ph in self._policyholders}
        self._ph_by_policy = {
            ph["policy_number"].upper(): ph for ph in self._policyholders
        }

        self._claims_by_party: dict[str, list] = {}
        for claim in self._claims:
            self._claims_by_party.setdefault(claim["party_id"], []).append(claim)

        self._claims_by_id = {c["case_id"]: c for c in self._claims}

        self._reps_by_buyer: dict[str, list] = {}
        for rep in self._representatives:
            self._reps_by_buyer.setdefault(rep["buyer_party_id"], []).append(rep)

    # -------------------------------------------------------- Queries

    def get_all_policyholders(self):
        return self._policyholders

    def get_policyholder_by_id(self, party_id):
        return self._ph_by_id.get(party_id)

    def get_policyholder_by_policy(self, policy_number):
        return self._ph_by_policy.get(policy_number.upper())

    def get_claims_for_party(self, party_id):
        return self._claims_by_party.get(party_id, [])

    def get_claim_by_id(self, case_id):
        return self._claims_by_id.get(case_id)

    def get_claim_schema(self):
        return self._claim_schema

    def get_document_guidelines(self):
        return self._document_guidelines

    def get_document_guidance(self, doc_name):
        guidance = self._document_guidelines.get("document_guidance", {})
        return guidance.get(doc_name, {}).get("en", "")

    def get_document_alternative_guidance(self, doc_name):
        alt = self._document_guidelines.get("document_alternative_guidance", {})
        specific = alt.get(doc_name, alt.get("default", {}))
        return specific.get("en", "")

    def get_case_type_guidance(self, case_type):
        ctg = self._document_guidelines.get("case_type_guidance", {})
        return ctg.get(case_type, {}).get("en", "")

    def get_followup_guidance(self):
        return self._document_guidelines.get("claim_followup_guidance", [])

    def get_followup_settings(self):
        return self._document_guidelines.get("claim_followup_settings", {})

    def get_followup_fallback(self):
        return (
            self._document_guidelines.get("claim_followup_fallback", {}).get("en", "")
        )

    def get_default_guidance(self):
        return self._document_guidelines.get("default_guidance", {}).get("en", "")

    def get_representatives_for_party(self, party_id):
        return self._reps_by_buyer.get(party_id, [])
