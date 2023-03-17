import logging
from configparser import ConfigParser

import requests


def escape_html(text: str):
    """
    Escapes all html characters in text
    :param str text:
    :rtype: str
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class TelegramStreamIO(logging.Handler):

    API_ENDPOINT = "https://api.telegram.org"
    MAX_MESSAGE_LEN = 4096
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s at %(funcName)s "
        "(line %(lineno)s):\n\n%(message)s"
    )

    def __init__(self, tg_configfile: str):
        super(TelegramStreamIO, self).__init__()
        config = ConfigParser()
        if not config.read(tg_configfile):
            raise FileNotFoundError(
                f"{tg_configfile} not found. " "Retry without --telegram-cred flag."
            )
        config = config["TELEGRAM"]
        token = config["token"]
        self.chat_id = config["chat_id"]
        self.url = f"{self.API_ENDPOINT}/bot{token}/sendMessage"

    @staticmethod
    def setup_logger(params):
        if not params.telegram_cred:
            return
        formatter = logging.Formatter(
            f"{params.exp_dir.name} %(asctime)s \n%(message)s"
        )
        tg = TelegramStreamIO(params.telegram_cred)
        tg.setLevel(logging.WARN)
        tg.setFormatter(formatter)
        logging.getLogger("").addHandler(tg)

    def emit(self, record: logging.LogRecord):
        """
        Emit a record.
        Send the record to the Web server as a percent-encoded dictionary
        """
        data = {
            "chat_id": self.chat_id,
            "text": self.format(self.mapLogRecord(record)),
            "parse_mode": "HTML",
        }
        try:
            requests.get(self.url, json=data)
            # return response.json()
        except Exception as e:
            logging.error(f"Failed to send telegram message: {repr(e)}")
            pass

    def mapLogRecord(self, record):
        """
        Default implementation of mapping the log record into a dict
        that is sent as the CGI data. Overwrite in your class.
        Contributed by Franz Glasner.
        """

        for k, v in record.__dict__.items():
            if isinstance(v, str):
                setattr(record, k, escape_html(v))
        return record
