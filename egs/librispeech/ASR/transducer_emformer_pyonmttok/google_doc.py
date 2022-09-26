from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os


def read_paragraph_element(element):
    """Returns the text in the given ParagraphElement.

    Args:
        element: a ParagraphElement from a Google Doc.
    """
    text_run = element.get("textRun")
    if not text_run:
        return ""
    return text_run.get("content"), element.get("endIndex", 1)


def read_structural_elements(elements):
    """Recurses through a list of Structural Elements to read a document's text where text may be
    in nested elements.

    Args:
        elements: a list of Structural Elements.
    """
    text = ""
    max_index = 1
    for value in elements:
        if "paragraph" in value:
            elements = value.get("paragraph").get("elements")
            for elem in elements:
                new_text, new_index = read_paragraph_element(elem)
                text += new_text
                max_index = max(max_index, new_index)
        elif "table" in value:
            # The text in table cells are in nested Structural Elements and tables may be
            # nested.
            table = value.get("table")
            for row in table.get("tableRows"):
                cells = row.get("tableCells")
                for cell in cells:
                    text += read_structural_elements(cell.get("content"))
        elif "tableOfContents" in value:
            # The text in the TOC is also in a Structural Element.
            toc = value.get("tableOfContents")
            text += read_structural_elements(toc.get("content"))
    return text, max_index


SCOPES = ["https://www.googleapis.com/auth/documents"]
SERVICE_ACCOUNT_FILE = "/cred/credentials.json"
DOCUMENT_ID = os.environ.get("GOOGLE_DOCUMENT_ID")


class GoogleDoc:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )

        self.service = build("docs", "v1", credentials=credentials)
        self.document_id = DOCUMENT_ID
        self.sent_index = 1

    def publish_text(self, partial_text_ready_for_submission):
        if not partial_text_ready_for_submission:
            return
        requests = [
            {
                "insertText": {
                    "endOfSegmentLocation": {},
                    "text": partial_text_ready_for_submission,
                }
            },
        ]
        self.service.documents().batchUpdate(
            documentId=self.document_id,
            body={"requests": requests + self.color_sent_request()},
        ).execute()

    def set_sent_index(self, sent_index):
        self.sent_index = sent_index

    def color_sent_request(self):
        if self.sent_index == 1:
            return []
        return [
            {
                "updateTextStyle": {
                    "range": {"startIndex": 1, "endIndex": self.sent_index},
                    "textStyle": {
                        "foregroundColor": {
                            "color": {
                                "rgbColor": {
                                    "blue": 1.0,
                                    "green": 0.0,
                                    "red": 0.0,
                                }
                            }
                        }
                    },
                    "fields": "foregroundColor",
                }
            }
        ]

    def get_text_from_document(self):
        res = (
            self.service.documents().get(documentId=self.document_id).execute()
        )
        text, max_index = read_structural_elements(res["body"]["content"])
        return text, max_index
