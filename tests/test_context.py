"""Память диалога: что сохраняется и что очищает /clear. Pinecone замокан."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hay_v2_bot.components.context import ChatMemory, should_save_message
from hay_v2_bot.components.document_store import delete_by_filter, user_file_filter, user_filter


def make_message(text="Меня зовут Иван", content_type="text", is_bot=False):
    return SimpleNamespace(
        content_type=content_type,
        text=text,
        message_id=42,
        date=1_700_000_000,
        from_user=SimpleNamespace(id=111, is_bot=is_bot),
        chat=SimpleNamespace(id=222),
    )


def test_user_text_is_saved():
    assert should_save_message(make_message("Я люблю горы")) is True


@pytest.mark.parametrize("text", ["/start", "/docs", "/clear", "  /cleardocs ", "", "   ", None])
def test_commands_and_empty_are_not_saved(text):
    assert should_save_message(make_message(text)) is False


@pytest.mark.parametrize("content_type", ["photo", "document", "voice"])
def test_non_text_is_not_saved(content_type):
    assert should_save_message(make_message("подпись", content_type=content_type)) is False


def test_bot_messages_are_not_saved():
    assert should_save_message(make_message(is_bot=True)) is False


def test_command_does_not_touch_pinecone():
    store = MagicMock()

    assert ChatMemory(store).save_user_message(make_message("/start"), [0.1]) is False
    store.write_documents.assert_not_called()


def test_clear_touches_only_chat_store():
    chat_store = MagicMock()
    chat_store.delete_by_filter.return_value = 4

    assert ChatMemory(chat_store).clear(111) == 4
    chat_store.delete_by_filter.assert_called_once_with(user_filter(111))


def test_delete_by_filter_fallback_for_old_store():
    store = MagicMock(spec=["filter_documents", "delete_documents"])
    store.filter_documents.side_effect = [[SimpleNamespace(id="a"), SimpleNamespace(id="b")], []]

    assert delete_by_filter(store, user_filter(1)) == 2
    store.delete_documents.assert_called_once_with(["a", "b"])


def test_user_file_filter():
    assert user_file_filter(5, "a.pdf") == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.user_id", "operator": "==", "value": "5"},
            {"field": "meta.file_name", "operator": "==", "value": "a.pdf"},
        ],
    }


def test_saved_with_meta_and_reused_embedding():
    pytest.importorskip("haystack")
    store = MagicMock()

    assert ChatMemory(store).save_user_message(make_message(), [0.1, 0.2]) is True

    doc = store.write_documents.call_args.args[0][0]
    assert doc.embedding == [0.1, 0.2]
    assert doc.meta == {
        "user_id": "111",
        "chat_id": "222",
        "timestamp": 1_700_000_000,
        "role": "user",
        "source": "chat",
    }
