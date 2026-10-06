"""
Email Generator — formats the POST_PROCESS email summary.
"""


class EmailGenerator:
    """Formats the structured email data into readable text."""

    @staticmethod
    def format_email(email_data: dict) -> str:
        parts = [
            f"Subject: {email_data.get('subject', 'Your Insurance Claim Summary')}",
            "",
            email_data.get("greeting", "Dear Valued Customer,"),
            "",
            "── Discussion Summary ──",
            email_data.get("discussion_summary", ""),
            "",
            "── Claim Status ──",
            email_data.get("claim_status", ""),
            "",
            "── Next Steps ──",
        ]

        for i, step in enumerate(email_data.get("next_steps", []), 1):
            parts.append(f"  {i}. {step}")

        parts.extend([
            "",
            email_data.get(
                "closing",
                "Thank you for contacting us. If you have further questions, "
                "please don't hesitate to call back.",
            ),
        ])

        return "\n".join(parts)
